import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
import torch
# from Data.CVC_ClinicDB.datasetCVC import Dataset

from glob import glob

import albumentations as A
from albumentations.core.composition import Compose, OneOf

def load_dataACDC(args):
    from Data.ACDC.datasetACDC import Dataset 
    # t = transforms.Compose([transforms.Resize((256, 256)),
    #                         transforms.ToTensor(),])
    # t = A.Compose([A.Resize(width=256, height=256),A.ChannelDropout(p=0.3),
    # A.RandomBrightnessContrast(p=0.3),A.ColorJitter(p=0.3),])
    t = A.Compose([A.Resize(width=256, height=256),])
    train_transform = Compose([
        A.RandomRotate90(),
        A.Flip(),
        OneOf([A.HueSaturationValue(),A.RandomBrightness(),A.RandomContrast(),], p=1),#按照归一化的概率选择执行哪一个
        A.Resize(256, 256),
        A.Normalize(),
    ])
    # train_transform = Compose([
    #     A.Resize(256, 256),
    #     A.Normalize(),
    # ])
    val_transform = Compose([
        A.Resize(256, 256),
        A.Normalize(),
    ])
    # imgs_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/images/*')
    # imgs_path.sort()
    # labels_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/masks/0/*')
    # labels_path.sort()
    
    
    dataPath = 'ACDC'
    trainPicFormat = '.jpg'
    testPicFormat = '.png'
    pathFile = '../Data'
    ImagePath = "image"
    MaskPath = "label"

    #  
    # import sys
    # sys.path.append('../')
    
    Path = os.path.join(pathFile, dataPath, ImagePath, '*' + trainPicFormat)

    imgNames = glob(Path)
    print(Path)
    imgNames = [os.path.splitext(os.path.basename(p))[0] for p in imgNames]
    length = int(len(imgNames))
    print(length)
    img_ids = imgNames
    args.seed = 42
    train_datapath,valid_datapath = train_test_split(img_ids,test_size=0.2,random_state = args.seed)
    # test_datapath,valid_datapath = train_test_split(other_datapath,test_size=0.5,random_state = 42)
    print("trainLen:",len(train_datapath),"| testLen:",0,"| validLen:",len(valid_datapath))

    BACH_SIZE=args.batch_size
    train_dataset = Dataset(
        img_ids=train_datapath,
        img_dir=os.path.join(pathFile, dataPath, ImagePath),
        mask_dir=os.path.join(pathFile, dataPath, MaskPath),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= trainPicFormat,
        mask_ext= testPicFormat,
        # num_classes=config['num_classes'],
        num_classes= args.num_classes ,
        transform=train_transform)
    # test_dataset = Dataset(
    #     img_ids=test_datapath,
    #     img_dir=os.path.join(pathFile, dataPath, ImagePath),
    #     mask_dir=os.path.join(pathFile, dataPath, MaskPath),
    #     # img_ext=config['img_ext'],
    #     # mask_ext=config['mask_ext'],
    #     img_ext= trainPicFormat,
    #     mask_ext= testPicFormat,
    #     # num_classes=config['num_classes'],
    #     num_classes= args.num_classes ,
    #     transform=val_transform)
    val_dataset = Dataset(
        img_ids=valid_datapath,
        img_dir=os.path.join(pathFile, dataPath, ImagePath),
        mask_dir=os.path.join(pathFile, dataPath, MaskPath),
        img_ext= trainPicFormat,
        mask_ext= testPicFormat,
        num_classes= args.num_classes ,
        transform=val_transform)
    "----------------------------"
    trainset = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=True,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=True)

    # testset = torch.utils.data.DataLoader(
    #     test_dataset,
    #     # batch_size=config['batch_size'],
    #     batch_size= BACH_SIZE,
    #     shuffle=True,
    #     # num_workers=config['num_workers'],
    #     num_workers= 0,
    #     drop_last=True)

    validset = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=False,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=False)
    return trainset, 0, validset

def load_dataCVC(args):
    from Data.CVC_ClinicDBNew.datasetCVC import Dataset
    # t = transforms.Compose([transforms.Resize((256, 256)),
    #                         transforms.ToTensor(),])
    # t = A.Compose([A.Resize(width=256, height=256),A.ChannelDropout(p=0.3),
    # A.RandomBrightnessContrast(p=0.3),A.ColorJitter(p=0.3),])
    t = A.Compose([A.Resize(width=256, height=256),])
    train_transform = Compose([
        A.RandomRotate90(),
        A.Flip(),
        OneOf([A.HueSaturationValue(),A.RandomBrightness(),A.RandomContrast(),], p=1),#按照归一化的概率选择执行哪一个

        A.Resize(256, 256),
        A.Normalize(),
    ])
    val_transform = Compose([
        A.Resize(256, 256),
        A.Normalize(),
    ])
    # imgs_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/images/*')
    # imgs_path.sort()
    # labels_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/masks/0/*')
    # labels_path.sort()
    if args.dataset == "CVC":
        dataPath = 'CVC_ClinicDBNew'
        trainPicFormat = '.tif'
        testPicFormat = '.tif'
        pathFile = '../Data'
        ImagePath = "Original"
        MaskPath = "Ground Truth"
    elif args.dataset == "ACDC":
        dataPath = 'ACDC'
        trainPicFormat = '.jpg'
        testPicFormat = '.png'
        pathFile = '../Data'
        ImagePath = "image"
        MaskPath = "label"

    #  
    # import sys
    # sys.path.append('../')
    
    Path = os.path.join(pathFile, dataPath, ImagePath, '*' + trainPicFormat)

    imgNames = glob(Path)
    print(Path)
    imgNames = [os.path.splitext(os.path.basename(p))[0] for p in imgNames]
    length = int(len(imgNames))
    print(length)
    img_ids = imgNames
    # val_img_ids   = img_ids[int(length*0.8):]
    # img_ids = img_ids[:int(len(img_ids)*0.8)]
    # kf
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # trainL=[]
    # testL=[]
    # for train_index, test_index in kf.split(imgNames):
    #     trainL.append([imgNames[i] for i in train_index])
    #     testL.append([imgNames[i] for i in test_index])
    # # img_ids =trainL[kfold]
    # img_ids =trainL[args.kf]
    # val_img_ids = testL[args.kf]
    
    # train_id1,train_id2_3 = train_test_split(img_ids,test_size=0.66,random_state = 42)
    # train_id2,train_id3   = train_test_split(train_id2_3,test_size=0.5,random_state = 42)
    # if client == "train":
    #     train_img_ids = img_ids
    # elif client == "client1":
    #     # train_img_ids, _ = train_test_split(img_ids, test_size=0.8,random_state=16)
    #     train_img_ids = train_id1
    # elif client == "client2":
    #     train_img_ids = train_id2
    # elif client == "client3":
    #     train_img_ids = train_id3
    # elif client == "trainBig":
    #     train_img_ids,_= train_test_split(img_ids,test_size=0.90,random_state = 42)
    args.seed = 42
    train_datapath,other_datapath = train_test_split(img_ids,test_size=0.2,random_state = args.seed)
    test_datapath,valid_datapath = train_test_split(other_datapath,test_size=0.5,random_state = 42)
    print("trainLen:",len(train_datapath),"| testLen:",len(test_datapath),"| validLen:",len(valid_datapath))

    BACH_SIZE=args.batch_size
    train_dataset = Dataset(
        img_ids=train_datapath,
        img_dir=os.path.join(pathFile, dataPath, ImagePath),
        mask_dir=os.path.join(pathFile, dataPath, MaskPath),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= trainPicFormat,
        mask_ext= testPicFormat,
        # num_classes=config['num_classes'],
        num_classes= 1 ,
        transform=train_transform)
    test_dataset = Dataset(
        img_ids=test_datapath,
        img_dir=os.path.join(pathFile, dataPath, ImagePath),
        mask_dir=os.path.join(pathFile, dataPath, MaskPath),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= trainPicFormat,
        mask_ext= testPicFormat,
        # num_classes=config['num_classes'],
        num_classes= 1 ,
        transform=val_transform)
    val_dataset = Dataset(
        img_ids=valid_datapath,
        img_dir=os.path.join(pathFile, dataPath, ImagePath),
        mask_dir=os.path.join(pathFile, dataPath, MaskPath),
        img_ext= trainPicFormat,
        mask_ext= testPicFormat,
        num_classes= 1 ,
        transform=val_transform)
    "----------------------------"
    trainset = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=True,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=True)

    testset = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=True,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=True)

    validset = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=False,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=False)
    return trainset, testset, validset