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
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils import DiceLoss
from torchvision import transforms

from _UnifiedModule import dice_coef,iou_score
from utils import DiceLoss,test_single_volume
sys.path.append('../../')
from model.transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from _UnifiedModule import get_args


def tester_synapseN(args,model,test_save_path=None):
    from Data.Synapse.dataset_synapse import Synapse_dataset, RandomGenerator

    if args.cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif args.cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif args.cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:2")

    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
                            #    transform=transforms.Compose(
                            #        [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of test set is: {}".format(len(db_test)))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    model.to(DEVICE)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(args,image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance , mean_hd95

if __name__ == "__main__":

    args = get_args()

    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if   args.cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif args.cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif args.cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:2")

    
    dataset_config = {
        'Synapse': {
            # 'Dataset': Synapse,
            'volume_path': '../Data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    # args.exp =  str(args.model) + '_' + dataset_name + str(args.img_size)
    args.exp = args.model+'_'+args.kd+args.Ukd+"_"+ dataset_name + str(args.img_size)
    # print(args.exp)
    snapshot_path = "model_Synapse/{}/{}".format(args.exp, args.model+"_"+args.kd)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC': # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.client) if args.client != 'train' else snapshot_path

    # if args.kd == "csf" or args.kd == "abf" or args.kd == "allFeature":
    #     snapshot_path = snapshot_path + '_KD:'+str(args.kd)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    # config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    # from model.simpleUnet import simpleUnet as simpleUnet
    # net = simpleUnet(num_classes=config_vit.n_classes,num_channels=1).to(DEVICE)

    
    
    if args.model == "simpleUnet":#3Layer-->118642
        from model.simpleUnet import simpleUnet as simpleUnet
        net = simpleUnet(num_classes=config_vit.n_classes,num_channels=1).to(DEVICE)
    elif args.model == "resUnet":#13040770
        from model.Unet_model.ResUnet.res_unet import ResUnet as ResUnet
        net = ResUnet(num_classes=config_vit.n_classes,num_channels=1).to(DEVICE)
    elif args.model == "transUnet":#105322146
        # config_vit = CONFIGS_ViT_seg[args.vit_name]
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
        net.load_from(weights=np.load('../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    elif args.model == "transUnetStu" or args.model == "transUnetKD":
        # config_vit = CONFIGS_ViT_seg[args.student_vit_name]
        # net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes,upsampling=2).to(DEVICE)
        net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)
    elif args.model == "swimUnet":
        from model.SwinUnet.config import get_config
        from model.SwinUnet.vision_transformer import SwinUnet as swinUnet
        config = get_config(args)
        net = swinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(DEVICE)
    else:
        print("Not Find the Net!")

    # if args.n_gpu > 1 and args.client == "train":
    #     net = nn.DataParallel(net ,device_ids=[1,2])
    if args.kd != "response" or args.kd != "no":
        from kd.utils_kd import U_KnowledgeDistillation
        net = U_KnowledgeDistillation(args=args,model=net,DEVICE=DEVICE)
        net = nn.DataParallel(net ,device_ids=[1,2])
    
    # snapshot_path = "../model/transUnet_Synapse224/transUnet_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224"
    # snapshot_path = "model/swimUnet_Synapse224/swimUnet_Train_pretrain_R50-ViT-B_16_skip3_epo150_bs32_lr0.05_224_s42"
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.num_epochs))
    print("testSavedModel:",snapshot)
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]
    
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # if args.is_savenii:
    args.test_save_dir = '../predictions'
    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
    # print("testSavedModel:",test_save_path)
    os.makedirs(test_save_path, exist_ok=True)

    # else:
        # test_save_path = None
    tester_synapseN(args, net, test_save_path)