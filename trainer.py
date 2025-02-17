import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils import DiceLoss
from torchvision import transforms

from _UnifiedModule import dice_coef,iou_score
from utils import DiceLoss
from utils import calculate_metric_percase as HD
from data_frame_utils import EarlyStopping

def train_DataSplit(args):
    from sklearn.model_selection import train_test_split
    slice_name = open(os.path.join(args.list_dir, 'train.txt')).readlines()
    print("TrainNum:",len(slice_name))
    train_id1,train_id2_3 = train_test_split(slice_name,test_size=0.66,random_state = 42)
    train_id2,train_id3   = train_test_split(train_id2_3,test_size=0.5,random_state = 42)
    if args.client == "train":
        slice_nameSplit = slice_name
    elif args.client == "client1":
        # train_img_ids, _ = train_test_split(img_ids, test_size=0.8,random_state=16)
        slice_nameSplit = train_id1
    elif args.client == "client2":
        slice_nameSplit = train_id2
    elif args.client == "client3":
        slice_nameSplit = train_id3
    return slice_nameSplit

# def calculate_flops_and_params(model, input):
#     # 创建一个示例输入 img
#     flops =0
#     for module in model.modules():
#         if isinstance(module, nn.Conv2d):
#             flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * input.size(2) * input.size(3)
#             input = module(input)
#         elif isinstance(module, nn.Linear):
#             flops += module.in_features * module.out_features
#             input = module(input.view(input.size(0), -1))
#     # 计算模型参数数量
#     params = sum(p.numel() for p in model.parameters())
#     return flops, params

def trainer_synapseN(args,model,snapshot_path):
    from Data.Synapse.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations
    # print("root_path:",args.root_path)
    # print("list_dir:",args.list_dir)
    #Split Clients' Dataset
    slice_name = train_DataSplit(args=args)
    

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",splitData=slice_name,#"train"
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)#num_workerw = 0-->8
    
    if args.cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif args.cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif args.cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:1")

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model ,device_ids=[1,2,3]) 
    # torch.backends.cudnn.benchmark=False #get CUDNN_STATUS_INTERNAL_ERROR
    model.train()
    # ce_loss = CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    epochs = args.num_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    
    # history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    history = {'train_loss': [], 'train_IoU': []}
    # early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        IoU = 0
        model.train()
        for i, data in enumerate(tqdm(trainloader)):
            img, mask = data['image'],data['label']
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            e,f,outputs = model(img)
            print(f'img:{img.shape}, mask:{mask.shape},predict:{outputs.shape}')

            # print(mask.shape)

            # 估算 FLOPs
            # flops, params = calculate_flops_and_params(model,img)

            # flops, params = clever_format([flops, params], "%.3f")
            # print(f"FLOPs: {flops},Params:{params}")

            # from torchsummary import summary
            # summary(model, (32, 1, 224, 224))

            # from torchstat import stat
            # stat(model,(32, 1, 224, 224))

            from thop import profile
            flops, params = profile(model, inputs=(img,))
            print(f'flop:{flops} | params:{params}')
            print("Params:",sum(p.numel() for p in model.parameters() if p.requires_grad))
            # transStu:6678009 105277081
            #flop:75477549056.0 | params:2411481.0
            #flop:791291074560.0 | params:93231705.0
            # for fea in f:
            #     print("fea_size: ", fea.shape)
            loss = dice_loss(outputs, mask.float(),softmax = True)
            running_loss += loss.item() * img.size(0)
            # IoU += iou_score(outputs,mask[:].float()).sum().item()
            # Dice,HD95 = HD(outputs, mask.float()) 
            IoU = 0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / len(trainloader.dataset)
        train_IoU = IoU / len(trainloader.dataset)
        
        save_interval = 10
        if epoch > int(epochs / 2) and (epoch+1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch >= epochs:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

        history['train_loss'].append(train_loss)
        writer.add_scalar("Training/Train_loss", train_loss, epoch)
        writer.add_scalar("Training/IoU", train_IoU, epoch)
        history['train_IoU'].append(train_IoU)

        logging.info(f'Epoch: {epoch}/{epochs} | Validation loss: {train_loss} | Validation Features: {train_IoU}')
        print(f'Epoch: {epoch}/{epochs} | Validation loss: {train_loss} | Validation Features: {train_IoU}')
        model.eval()
    writer.close()
    
    return "Training Finished!"

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)



def trainer_synapseKD(args,model,modelTea,snapshot_path):
    from Data.Synapse.dataset_synapse import Synapse_dataset, RandomGenerator
    from kd.ConvF import getSameFeatureConv3, loss_Softmax , loss_fn
    from FKD2Med.kd.CSFOld import build_kd_trans, hcl
    from kd.utils_kd import U_KnowledgeDistillation,KDloss

    # from FeatureKD_Unet.train_mySynapse.featureKD.kd.ABF import build_review_kd

    if args.cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif args.cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif args.cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:1")

    model = U_KnowledgeDistillation(args=args,model=model,DEVICE=DEVICE)
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size 
    # max_iterations = args.max_iterations
    slice_name = train_DataSplit(args=args)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",splitData=slice_name,#"train"
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)#num_workerw = 0-->8

    # torch.backends.cudnn.benchmark=False #get CUDNN_STATUS_INTERNAL_ERROR
    model.train()
    modelTea.eval()
    # ce_loss = CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    epochs = args.num_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    
    weight = torch.ones(num_classes).to(DEVICE)
    criterion = CrossEntropyLoss2d(weight).to(DEVICE)

    # history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    history = {'train_loss': [], 'train_IoU': []}
    # early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        running_lossFeature = 0
        IoU = 0
        for i, data in enumerate(tqdm(trainloader)):
            img, mask = data['image'],data['label']
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            # print(img.shape, mask.shape)
            e,f,outputs = model(img)
            
            from thop import profile
            flops, params = profile(model, inputs=(img,))
            print(f'flop:{flops} | params:{params} | Params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
            #student    flop:75477549056.0  | params:2411481.0 | Params:6678009
            #response   flop:75477549056.0  | params:2411481.0 | Params:6678009
            #allFeature flop:134516637696.0 | params:3581785.0 | Params:7848313
            #abf        flop:134584074240.0 | params:3583327.0 | Params:7849855
            #csf        flop:134534762496.0 | params:3619609.0 | Params:7898425
            #SKembed    flop:142087219200.0 | params:3780025.0 | Params:8047321
            with torch.no_grad():
                eT,fT,outputT = modelTea(img)
                if args.Ukd == "Decode":fT = fT[3:]
                elif args.Ukd == "Encode":fT = fT[:3]
                elif args.Ukd == "EnDe":pass
                else: pass
                # fT = fT[1:]
            # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            # for fea in f:
            #     print("stu:",fea.shape)
            # for fea in fT:
            #     print("tea:",fea.shape)
            # outputs = outputs.squeeze(1)
            # print(torch.max(predictions))
            # loss = criterion(outputs, mask.float())
            loss_logits = dice_loss(outputs, mask.float(),softmax = True)
            # loss_logits = criterion(outputs, mask)
            running_loss += loss_logits.item() * img.size(0)
            # IoU += iou_score(outputs,mask[:].float()).sum().item()
            # Dice,HD95 = HD(outputs, mask.float()) 
            IoU = 0
            loss_feature,loss = KDloss(args=args,loss_logits=loss_logits,f=f,fT=fT,x=outputs,xT=outputT)
            running_lossFeature += loss_feature.item() * img.size(0)
            # print(running_loss)
            # print(IoU)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / len(trainloader.dataset)
        train_IoU = running_lossFeature / len(trainloader.dataset)
        
        save_interval = 10
        if epoch > int(epochs / 2) and (epoch+1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch >= epochs:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

        history['train_loss'].append(train_loss)
        writer.add_scalar("Training/Train_loss", train_loss, epoch)
        writer.add_scalar("Training/IoU", train_IoU, epoch)
        history['train_IoU'].append(train_IoU)

        logging.info(f'Epoch: {epoch}/{epochs} | Validation loss: {train_loss} | Validation Features: {train_IoU}')
        print(f'Epoch: {epoch}/{epochs} | Validation loss: {train_loss} | Validation Features: {train_IoU}')
    model.eval()
    writer.close()
    return "Training Finished!"


