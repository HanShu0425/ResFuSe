import argparse
import logging
import os
import random
import numpy as np
import torch.nn as nn
import torch
# import torch.backends.cudnn as cudnn
import sys
sys.path.append('../')
from model.transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from torch.utils.data import DataLoader
from trainer import trainer_synapseN
# export PYTHONUNBUFFERED=1 #cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
from _UnifiedModule import get_args


if __name__ == "__main__":
    
    args = get_args()
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    if args.cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif args.cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif args.cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:2")
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../Data/Synapse/train_npz',
            'list_dir': 'lists/lists_Synapse',
            'num_classes': 9,
        },

    }
    if args.batch_size != 24 and args.batch_size % 6 == 0 and args.model != "swimUnet":
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = args.model+'_' + dataset_name + str(args.img_size)

    snapshot_path = "model_Synapse/{}/{}".format(args.exp, args.model+"_Train")
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.client) if args.client != 'train' else snapshot_path
    # print(snapshot_path)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    
    # if args.model == "simpleUnet":
    #     from model.simpleUnet import simpleUnet as simpleUnet
    #     net = simpleUnet(num_classes=args.num_classes,num_channels=1).to(DEVICE)
    # elif args.model == "transUnet":
    #     net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
    #     net.load_from(weights=np.load('../../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    # elif args.model == "transUnetStu":
    #     # config_vit = CONFIGS_ViT_seg[args.student_vit_name]
    #     # net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes,upsampling=2).to(DEVICE)
    #     net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)

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
    elif args.model == "swimUnet":
        from model.SwinUnet.config import get_config
        from model.SwinUnet.vision_transformer import SwinUnet as swinUnet
        config = get_config(args)
        net = swinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(DEVICE)

    elif args.model == "transUnetStu":
        # config_vit = CONFIGS_ViT_seg[args.student_vit_name]
        # net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes,upsampling=2).to(DEVICE)
        net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)
    else:
        print("Not Find the Net!")

    # if args.n_gpu > 1 and args.client == "train":
    #     net = nn.DataParallel(net ,device_ids=[1,2])
    trainer = {'Synapse': trainer_synapseN}
    trainer[dataset_name](args, net, snapshot_path)