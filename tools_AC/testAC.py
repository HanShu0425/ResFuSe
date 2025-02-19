import os
import time
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from data_frame_utils import get_file_row, iou_pytorch, dice_pytorch, BCE_dice, EarlyStopping,LogCoshBDLoss73
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch

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
from medpy import metric
import SimpleITK as sitk

# def calculate_metric_percase(pred, gt):#HD
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum()>0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum()==0:
#         return 1, 0
#     else:
#         return 0, 0

def calculate_iou(y_true, y_pred):
    """
    计算单类别的IOU
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / (union + 1e-10)  # 避免除零
    return iou

def calculate_metric_percase(pred, gt):#dice + HD + asd + iou
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd  = metric.binary.asd(pred, gt)
        iou  = calculate_iou(pred, gt)
        return dice, hd95, asd, iou
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 0, 0
    else:
        return 0, 0, 0, 0
    
def  test_single(args, image, label, net, test_path,):
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image_Old = image.cpu().detach().numpy()
    image, label = image.squeeze(0).cpu().detach().numpy(), label.cpu().detach().numpy()
    # print("shape:",image.shape,label.shape)#shape: (3, 256, 256) (256, 256)
    prediction = np.zeros_like(label)
    # print("Pre_shape:",prediction.shape)#256,256
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        # x, y = slice.shape[0], slice.shape[1]
        input = torch.from_numpy(slice).unsqueeze(0).float().to(DEVICE)
        # print(input.shape)
        net.eval()
        with torch.no_grad():
                e,f,outputs = net(input)
                # print(outputs.shape)#1，3，256，256
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                # print("1_",out.shape)#(224, 224)
                pred = out
                # print(out.shape)#(224, 224)
                # print("1",pred.shape)
                # prediction[ind] = pred
                prediction[ind] = pred
                
    metric_list = []
    # for i in range(0, args.num_classes):
    #     # print(f'{i}_pre_shape:{prediction.shape} | label_shape:{label.shape}')
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    for i in range(0, args.num_classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        # logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        # logging.info('Mean class %d mean_asd  %f mean_iou  %f' % (i, metric_list[i-1][2], metric_list[i-1][3]))
    
    if test_path is not None:
        # image_uint8 = (image.astype(np.float32) * 255).astype(np.uint8)
        # prediction_uint8 = (prediction.astype(np.float32) * 255).astype(np.uint8)
        # label_uint8 = (label.astype(np.float32) * 255).astype(np.uint8)

        # img_itk = sitk.GetImageFromArray(image.astype(np.float32) * 255)
        # prd_itk = sitk.GetImageFromArray(prediction_uint8)
        # lab_itk = sitk.GetImageFromArray(label_uint8)
        img_itk = sitk.GetImageFromArray(image_Old.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        z_spacing = 1
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))#.nii.gz
        sitk.WriteImage(prd_itk, test_path + '/'+str(ind) + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_path + '/'+ str(ind) + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_path + '/'+ str(ind) + "_gt.nii.gz")
        # sitk.WriteImage(prd_itk, test_path + '/'+str(ind) +"_pred.png")
        # sitk.WriteImage(img_itk, test_path + '/'+str(ind) + "_img.png")
        # sitk.WriteImage(lab_itk, test_path + '/'+str(ind) + "_gt.png")


    return metric_list

def test_run(args, model, test_path, test_loader, loss_fn):
    
    # DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    model.eval()
    # model.to(DEVICE)
    metric_list = 0.0
    logging.info("{} iterations per epoch. {} max iterations ".format(len(test_loader), 1))


    with torch.no_grad():   
        # running_IoU = 0
        # running_dice = 0
        # running_test_loss = 0
        for i, data in enumerate(test_loader):
            print(f'{i}_test')
            img, mask = data
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            metric_i = test_single(args, img, mask, net, test_path,)
            metric_list += np.array(metric_i)
            logging.info('idx %d mean_dice %f mean_hd95 %f' % (i, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            logging.info('idx %d mean_asd %f mean_iou %f' % (i, np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))

            # running_dice += dice_pytorch(predictions, mask).sum().item()
            # running_IoU += iou_pytorch(predictions, mask).sum().item()
            # loss = loss_fn(predictions, mask)
            # running_test_loss += loss.item() * img.size(0)
    metric_list = metric_list / len(test_loader)
    for i in range(1, args.num_classes+1):
        logging.info('Mean class %d mean_dice %f, mean_hd95 %f, mean_asd  %f, mean_iou  %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        # logging.info('Mean class %d mean_asd  %f mean_iou  %f' % (i, metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mASD = np.mean(metric_list, axis=0)[2]
    mIoU = np.mean(metric_list, axis=0)[3]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing performance in best val model: mean_dice %f mean_hd95 %f mean_asd  %f mean_iou  %f' % ( performance, mean_hd95,  mASD, mIoU))
    # logging.info('Testing performance in best val model: mean_asd  %f mean_iou  %f' % ( mASD, mIoU))

    # return performance , mean_hd95, mASD, mIoU

    # test_loss = running_test_loss / len(test_loader.dataset)
    # test_dice = running_dice / len(test_loader.dataset)
    # test_IoU = running_IoU / len(test_loader.dataset)
    # logging.info(f'| Test loss: {test_loss} | Test Mean IoU: {test_IoU} '
    #     f'| Test Dice coefficient: {test_dice}')




if __name__ == "__main__":

    args = get_args()
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    # Tensorboard
    writer = SummaryWriter("tensorboard_logs")

    args.batch_size = 12#24 8,12,16   12 is best
    args.img_size = 256
    train_loader, test_loader, valid_loader, = load_dataACDC(args)

    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    args.num_classes = 4
    config_vit.n_classes =  int(args.num_classes)#1
    config_vitStu = CONFIGS_ViT_seg['R50-ViT-B_16_student']
    config_vitStu.n_classes =  int(args.num_classes)#1
    config_vit.n_skip = 3

    
    if args.model == "transUnet":#105322146
        # config_vit = CONFIGS_ViT_seg[args.vit_name]
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(DEVICE)
        net.load_from(weights=np.load('../model/transUnet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    elif args.model == "transUnetStu" or args.model == "transUnetKD":
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
        net = torch.nn.DataParallel(net ,device_ids=[1,2])
        net = U_KnowledgeDistillation(args=args,model=net,DEVICE=DEVICE)
        
    

    args.base_lr = 0.005
    from torch.optim import Adam
    optimizer = Adam(net.parameters(), lr=args.base_lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    epochs = int(args.num_epochs)
    # print("num_epochs:",epochs)
    # lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=10,factor=0.1, 
    #                                  verbose=1, min_lr=1e-5)
    # from torch.optim import lr_scheduler
    # lr_scheduler = lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=500, eta_min=1e-5)
    
    from NormalUtils import DiceLoss
    loss_fn = DiceLoss(args.num_classes)

    train_path = "model_ACDC/{}/{}".format(args.model,args.kd)
    train_path = train_path+'_bs'+str(args.batch_size)
    train_path = train_path+ '_lr' + str(args.base_lr) if args.base_lr != 0.01 else train_path
    train_path = train_path+ '_'+str(args.img_size)
    train_path = train_path+ '_s'+str(args.seed) if args.seed!=1234 else train_path
    train_path = train_path+ '_'+str(args.client) if args.client != 'train' else train_path
    # train_path = "{}/{}".format(train_path,args.Ukd) if (args.kd != "Train" or args.kd != "response") else train_path

    trainFile = os.path.join(train_path, 'best_model_.pth')
    # trainFile = os.path.join(train_path, 'best_model.pth') 
    net.load_state_dict(torch.load(trainFile))
    test_path = train_path
    logging.basicConfig(filename=test_path + '/'+"test.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(trainFile))
    logging.info(str(args))
    
    test_run(args, net, test_path, valid_loader, loss_fn)