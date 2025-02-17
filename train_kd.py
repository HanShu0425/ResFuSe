import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import sys
sys.path.append('../')
from model.transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapseKD
from data_frame_utils import EarlyStopping
# export PYTHONUNBUFFERED=1 #cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
from _UnifiedModule import get_args
from trainer import train_DataSplit


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
    dataset_config = {#train+test
        'Synapse': {
            'root_path': '../Data/Synapse/train_npz',
            'volume_path': '../Data/Synapse/test_vol_h5',
            'z_spacing': 1,
            'list_dir': 'lists/lists_Synapse',
            'num_classes': 9, 
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = args.model+'_'+args.kd+args.Ukd+"_"+ dataset_name + str(args.img_size)

    snapshot_path = "model_Synapse/{}/{}".format(args.exp, args.model+"_"+args.kd)
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
    # snapshot_path = snapshot_path + '_kdPath'+str(args.Ukd) if args.kd!= "response" else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.client) if args.client != 'train' else snapshot_path

    expTea = 'transUnet_' + dataset_name + str(args.img_size)
    snapshot_pathTea = "model_Synapse/{}/{}".format(expTea, "transUnet_Train")
    snapshot_pathTea = snapshot_pathTea + '_pretrain' if args.is_pretrain else snapshot_pathTea
    snapshot_pathTea += '_' + args.vit_name
    snapshot_pathTea = snapshot_pathTea + '_skip' + str(args.n_skip)
    snapshot_pathTea = snapshot_pathTea + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_pathTea
    snapshot_pathTea = snapshot_pathTea+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_pathTea
    snapshot_pathTea = snapshot_pathTea + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_pathTea
    snapshot_pathTea = snapshot_pathTea +'_bs'+str(24)
    snapshot_pathTea = snapshot_pathTea + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_pathTea
    snapshot_pathTea = snapshot_pathTea + '_'+str(args.img_size)
    snapshot_pathTea = snapshot_pathTea + '_s'+str(args.seed) if args.seed!=1234 else snapshot_pathTea
    snapshot_pathTea = snapshot_pathTea + '_'+str(args.client) if args.client != 'train' else snapshot_pathTea
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
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
    # elif args.model == "transUnetKD":
    #     # config_vit = CONFIGS_ViT_seg[args.student_vit_name]
    #     # net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes,upsampling=2).to(DEVICE)
    #     net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)
    net = ViT_seg(CONFIGS_ViT_seg[args.student_vit_name],img_size=args.img_size, num_classes = args.num_classes).to(DEVICE)

    netTea = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
    netTea.load_from(weights=np.load('../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))

    if args.n_gpu > 1:
        # net = nn.DataParallel(net ,device_ids=[1,2])
        netTea = nn.DataParallel(netTea ,device_ids=[1,2])
    
    # snapshotTea = os.path.join(snapshot_pathTea, 'best_model.pth')
    # if not os.path.exists(snapshotTea): snapshotTea = snapshotTea.replace('best_model', 'epoch_150')
    # print("snapshot:",snapshot)
    netTea.load_state_dict(torch.load("../FeatureKD_Unet/model/transUnet_Synapse224/transUnet_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224_s42/epoch_99.pth"))
    # print("testSavedModel:",snapshotTea)
    # netTea.load_state_dict(torch.load(snapshotTea))    
    trainer = {'Synapse': trainer_synapseKD}
    trainer[dataset_name](args, net, netTea,snapshot_path)
