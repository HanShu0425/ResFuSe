import os
import time
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
from glob import glob
from data_frame_utils import get_file_row, iou_pytorch, dice_pytorch, BCE_dice, EarlyStopping,LogCoshBDLoss73
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import sys
sys.path.append('../')
# from Data.kaggle_3m.dataset_mri import MriDataset
from model.transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter
from _UnifiedModule import get_args
import logging
from data_load import load_dataACDC

import torch.optim as optim
from kd.utils_kd import U_KnowledgeDistillation,KDloss


def training_kd(args, train_path, writer, epochs, model,modelTea, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    logging.basicConfig(filename=train_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(lr_scheduler))
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2") 
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), epochs))

    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=200)

    model = U_KnowledgeDistillation(args=args,model=model,DEVICE=DEVICE)

    min_loss = 1
    for epoch in range(1, epochs + 1):
        running_loss = 0
        running_lossLogit = 0
        running_lossFeature = 0
        model.train()
        modelTea.eval()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            
            e,f,predictions = model(img)
            # for fea in f:
            #     print(f'feture_size:',{fea.shape})

            # from thop import profile
            # # flops, params = profile(model, inputs=(img.to(DEVICE),))
            # flops, params = 0,0
            # print(f'flop:{flops} | params:{params} | Params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

            with torch.no_grad():
                eT,fT,outputT = modelTea(img)
                if args.Ukd == "Decode":
                    if args.KDlayers == 4:
                        fT = fT[3:]
                    elif args.KDlayers == 3:
                        fT = fT[4:]
                    elif args.KDlayers == 2:
                        fT = fT[5:]
                    else:
                        print("Ablation Feature Layers Error!")
                elif args.Ukd == "Encode":fT = fT[:3]
                elif args.Ukd == "EnDe":pass
                else: pass

            
            #student    
            #response   
            #allFeature 
            #           
            #           

            #abf        
            #           
            #           

            #csf        
            #           
            #           

            #SKembed    
            #ResFuSe    
            #           
            #           

            # print(f'img:{img.shape}, mask:{mask.shape},predict:{predictions.shape}')
            # print(predictions.shape)
            # predictions = predictions.squeeze(1)
            # mask = mask.squeeze(1)
            # print(torch.sum(mask).item())
            # loss = loss_fn(predictions, mask)
            loss_logits = loss_fn(predictions, mask.float(),softmax = True)
            loss_feature,loss = KDloss(args=args,loss_logits=loss_logits,f=f,fT=fT,x=predictions,xT=outputT)

            # print(torch.max(predictions))
            running_lossLogit += loss_logits.item() * img.size(0)
            running_loss += loss.item() * img.size(0)
            running_lossFeature += loss_feature.item() * img.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        

        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                e,f,predictions = model(img)
                # predictions = predictions.squeeze(1)
                # mask = mask.squeeze(1)
                # running_dice += dice_pytorch(predictions, mask).sum().item()
                # running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask.float(),softmax = True)
                running_valid_loss += loss.item() * img.size(0)

        train_lossLogit = running_lossLogit / len(train_loader.dataset)
        feature_loss = running_lossFeature / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)

        history['train_loss'].append(train_lossLogit)
        writer.add_scalar("Training/Train_loss", train_lossLogit, epoch)
        writer.add_scalar("Training/Val_loss", val_loss, epoch)
        writer.add_scalar("Metric/Feature_loss", feature_loss, epoch)
        writer.add_scalar("Metric/Val_Dice", val_dice, epoch)


        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)
        logging.info(f'Epoch:{epoch}/{epochs} | TrainLogit_loss: {train_lossLogit} | Validation loss: {val_loss} | Feature_loss: {feature_loss} | Validation Mean IoU: {val_IoU} '
         f'| Validation Dice coefficient: {val_dice}')
        # print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
        #  f'| Validation Dice coefficient: {val_dice}')

        # lr_scheduler.step()
        # lr_scheduler.step(val_loss)

        # if early_stopping(val_loss, model):
        #     print("saveModel,epoch:",epoch)
        #     save_mode_path = os.path.join(train_path, 'epoch_' + str(epoch) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     # early_stopping.load_weights(model)
        #     break
        
        if epoch > int(epochs / 5) and val_loss < min_loss:
            min_loss = val_loss
            max_IoU = val_IoU
            max_dice = val_dice
            save_mode_path = os.path.join(train_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # save_interval = 20
        # if epoch > int(epochs / 2) and epoch % save_interval == 0:
        #     save_mode_path = os.path.join(train_path, 'epoch_' + str(epoch) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch >= epochs:
            save_mode_path = os.path.join(train_path, 'epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break
    model.eval()
    logging.info(f'best_model: min_loss: {min_loss} | IoU {max_IoU} | Dice {max_dice}')
    return history


if __name__ == "__main__":

    # csv_path = '../Data/kaggle_3m/kaggle_3m/data.csv'
    # files_dir = '../Data/kaggle_3m/kaggle_3m/'
    # file_paths = glob(f'{files_dir}/*/*[0-9].tif')
    # df = pd.read_csv(csv_path)
    # imputer = SimpleImputer(strategy="most_frequent")
    # df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    # filenames_df = pd.DataFrame((get_file_row(filename) for filename in file_paths), columns=['Patient', 'image_filename', 'mask_filename'])
    # df = pd.merge(df, filenames_df, on="Patient")
    # train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    # test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=42)
    # print(df)

    args = get_args()
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    # Tensorboard
    writer = SummaryWriter("tensorboard_logs")

    args.batch_size = 12#24 8,12,16   12 is best
    args.img_size = 256
    train_loader,  test_loader, valid_loader, = load_dataACDC(args)


    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes    =  int(args.num_classes)#1
    config_vitStu = CONFIGS_ViT_seg['R50-ViT-B_16_student']
    config_vitStu.n_classes =  int(args.num_classes)#1
    config_vit.n_skip = 3

    teacher = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
    # teacher = torch.nn.DataParallel(teacher ,device_ids=[1,2])
    teacher.load_from(weights=np.load('../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    
    
    
    if args.model == "transUnet":#105322146
        # config_vit = CONFIGS_ViT_seg[args.vit_name]
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
        net.load_from(weights=np.load('../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    elif args.model == "swimUnet":
        from model.SwinUnet.config import get_config
        from model.SwinUnet.vision_transformer import SwinUnet as swinUnet
        config = get_config(args)
        net = swinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(DEVICE)
    elif args.model == "transUnetStu":
        # net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes,upsampling=2).to(DEVICE)
        net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)
    else:
        print("Not Find the Net!")

    args.base_lr = 0.005
    optimizer = Adam(net.parameters(), lr=args.base_lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    epochs = int(args.num_epochs)
    # print("num_epochs:",epochs)
    # lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=10,factor=0.1, 
    #                                  verbose=1, min_lr=1e-5)
    from torch.optim import lr_scheduler
    lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=32, eta_min=1e-5)
    # from torch.optim import lr_scheduler
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    from NormalUtils import DiceLoss
    loss_fn = DiceLoss(args.num_classes)

    # loss_fn = BCE_dice
    # loss_fn = F.cross_entropy
    # from torch.nn.modules.loss import CrossEntropyLoss
    # loss_fn =CrossEntropyLoss()
    # loss_fn = LogCoshBDLoss73()
    train_path = "model_ACDC/{}/{}".format(args.model,args.kd)
    train_path = train_path+'_bs'+str(args.batch_size)
    train_path = train_path+ '_lr' + str(args.base_lr) if args.base_lr != 0.01 else train_path
    train_path = train_path+ '_'+str(args.img_size)
    train_path = train_path+ '_s'+str(args.seed) if args.seed!=1234 else train_path
    train_path = train_path+ '_'+str(args.client) if args.client != 'train' else train_path
    train_path = "{}/{}".format(train_path,args.Ukd) if args.kd != "Train" or args.kd != "response" else train_path
    # train_path = train_path+'_'+str(args.Ukd) if args.kd != "Train" or args.kd != "response" else train_path

    # expTea = 'transUnet_' + dataset_name + str(args.img_size)
    if args.n_gpu > 1:
        net = torch.nn.DataParallel(net ,device_ids=[1,2])
        teacher = torch.nn.DataParallel(teacher ,device_ids=[1,2])
    teacher.load_state_dict(torch.load("model_ACDC/transUnet/Train_bs12_lr0.005_256_s42/best_model_.pth"))
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    history = training_kd(args, train_path, writer, epochs, net, teacher, train_loader, test_loader, optimizer, loss_fn, lr_scheduler)