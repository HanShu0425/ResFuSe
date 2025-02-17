import argparse
import torchvision.transforms as transforms
# from DataSet import Dataset
from torch.utils import data
import torch
import glob
from glob import glob
import os
# import cv2
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

#tifPicShow
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
#unet++
import albumentations as A
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

#Go to the upper folder
import sys
sys.path.append('../')
# __all__ = [ 'BCEDiceLoss']


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, default = "",required=True, help="path to your train dataset")
    # #Train
    # parser.add_argument("--test", type=str, default = "",help="path to your test dataset")
    #Test
    parser.add_argument("--client", type=str, default="train", help="path to your data")
    parser.add_argument("--batch_size", type=int, default=32, help="path to your data")
    parser.add_argument("--cuda", type=str, default="cuda2", help="path to your data")
    parser.add_argument("--model", type=str, default="simpleUnet", help=" simpleUnet | resUnet | transUnet ")
    # parser.add_argument("--dataset", type=str, default="Chest", help="name to be appended to checkpoints")
    parser.add_argument('--dataset', type=str,default='Synapse', help='experiment_name')
    parser.add_argument("--picFormat",type=str, default=".png",help="name your datas' format")
    # parser.add_argument("--loss",type=str,default="crossentropy",
    #                     help="focalloss | iouloss | crossentropy",)#Loss
    parser.add_argument("--num_epochs", type=int, default=20, help="dnumber of epochs")
    parser.add_argument('--num_classes', default=9, type=int,help='number of classes')
    # parser.add_argument('--input', default=256, type=int,help='image width')
    parser.add_argument("--kd",type=str,default="csf",help="mid | softMax | CSF | Logits")
    parser.add_argument("--Ukd",type=str,default="Decode",help="Encode | Decode | EnDe")

    #tester
    parser.add_argument("--train",type=str,default="Train",help="Train | AllFeature")
    # #Epochs
    # parser.add_argument("--batch", type=int, default=1, help="batch size")
    
    #swimUnet
    # parser.add_argument('--cfg', type=str, required=False, default="../model/SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file' )
    
    # parser.add_argument(
    #         "--opts",
    #         help="Modify config options by adding 'KEY VALUE' pairs. ",
    #         default=None,
    #         nargs='+',
    #     )
    
    # parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    
    # parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
    #                     help='no: no cache, '
    #                             'full: cache all data, '
    #                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    
    # parser.add_argument('--use-checkpoint', action='store_true',
    #                     help="whether to use gradient checkpointing to save memory")
    # parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
    #                     help='mixed precision opt level, if O0, no amp is used')
    # parser.add_argument('--tag', help='tag of experiment')
    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # parser.add_argument('--throughput', action='store_true', help='Test throughput only'),
    #ours
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
    parser.add_argument('--seed', type=int,#1234
                    default=42, help='random seed')
    parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str,default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--student_vit_name', type=str,default='R50-ViT-B_16_student', help='select one vit model')
    parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
    
    
    return parser.parse_args()

def load_data(args):
    client = args.client
    dataset = args.dataset
    picFormatName =args.picFormat
    
    print('-----------------Data Loader Begining---------------------------')
    t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor(),])

    if dataset == 'Chest':
        dataPath = 'Chest_Xray_Masks_and_Labels'
    elif dataset == 'CVC':
        dataPath = 'CVC_ClinicDB'
    else:
        print('Dataset is Error!Not Choice')

    if picFormatName == '.png':
        picFormat = '.png'
    elif picFormatName == '.tif':
        picFormat = '.tif'
    else:
        print('Datasets format is Error!Not Choice')
    Path = '../Data'
    img_ids = glob(os.path.join(Path, dataPath, 'images', '*' + picFormat))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    length = int(len(img_ids))
    val_img_ids   = img_ids[int(length*0.8):]

    img_ids = img_ids[:int(len(img_ids)*0.8)]
    train_id1,train_id2_3 = train_test_split(img_ids,test_size=0.66,random_state = 42)
    train_id2,train_id3   = train_test_split(train_id2_3,test_size=0.5,random_state = 42)
    if client == "train":
        train_img_ids = img_ids
    elif client == "client1":
        # train_img_ids, _ = train_test_split(img_ids, test_size=0.8,random_state=16)
        train_img_ids = train_id1
    elif client == "client2":
        train_img_ids = train_id2
    elif client == "client3":
        train_img_ids = train_id3
    print(len(train_img_ids))
    print(len(val_img_ids))

    BACH_SIZE=4
    workers = 2

    train_transform = Compose([
        A.RandomRotate90(),
        A.Flip(),
        OneOf([A.HueSaturationValue(),A.RandomBrightness(),A.RandomContrast(),], p=1),#按照归一化的概率选择执行哪一个

        # A.Resize(config['input_h'], config['input_w']),
        A.Resize(256,256),
        A.Normalize(),
    ])
    val_transform = Compose([
        # A.Resize(config['input_h'], config['input_w']),
        A.Resize(256,256),
        A.Normalize(),
    ])
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(Path, dataPath, 'images'),
        mask_dir=os.path.join(Path, dataPath, 'masks'),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= picFormat,
        mask_ext= picFormat,
        # num_classes=config['num_classes'],
        num_classes= args.num_classes ,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(Path, dataPath, 'images'),
        mask_dir=os.path.join(Path, dataPath, 'masks'),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= picFormat,
        mask_ext= picFormat,
        # num_classes=config['num_classes'],
        num_classes= args.num_classes ,
        transform=val_transform)
    
    trainset = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=True,
        # num_workers=config['num_workers'],
        num_workers= workers,
        drop_last=True)
    print("Batch trainPics:",len(trainset))
    testset = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=False,
        # num_workers=config['num_workers'],
        num_workers= workers,
        drop_last=False)
    print("Batch testPics:",len(testset))
    # TrainDataSet = segDataset(img_paths = train_path,anno_paths = train_labels_path,transform = t)
    # TestDataSet  = segDataset(img_paths = test_path,anno_paths = test_labels_path,transform = t)
    # trainset = data.DataLoader(TrainDataSet,batch_size=BACH_SIZE,
    #                            shuffle=True, num_workers=0)#num_works =8
    # testset = data.DataLoader( TestDataSet, batch_size=BACH_SIZE,
    #                            shuffle=False, num_workers=0)#num_works =8
    return trainset, testset


def show_tif(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




#loss
def dice_loss(input, target):
    smooth = 1e-5
    input = torch.sigmoid(input)
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return dice_loss(input, target)


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return bce




class LogCoshBDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        bce = F.binary_cross_entropy_with_logits(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        x = bce + dice
        return x

#acc

# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)
def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # intersection = (output * target).sum()
    intersection = output.dot(target)

    return (intersection + smooth) / \
        (output.sum() + target.sum() - intersection + smooth)
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def ppv_ppv(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / (output.sum() + smooth)