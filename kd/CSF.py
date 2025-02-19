import torch
from torch import nn
import torch.nn.functional as F

class SK(nn.Module):#CSF
    def __init__(self, in_channel, mid_channel, out_channel, fuse, len=32, reduce=16):
        super(SK, self).__init__()
        len = max(mid_channel // reduce, len)
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            #https://github.com/syt2/SKNet
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(len),
                nn.ReLU(inplace=True)
            )
            self.fc1 = nn.Sequential(
                nn.Conv2d(mid_channel, len, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fcs = nn.ModuleList([])
            for i in range(2):
                self.fcs.append(
                    nn.Conv2d(len, mid_channel, kernel_size=1, stride=1)
                )
            self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1) 

    def forward(self, x, y=None, shape=None):
        x = self.conv1(x)
        if self.fuse:
            # print("x",x.shape)
            
            shape = x.shape[-2:]
            b = x.shape[0]
            y = F.interpolate(y, shape, mode="nearest")
            # print("y",y.shape)
            feas_U = [x,y]
            feas_U = torch.stack(feas_U,dim=1)
            # print("feas_U",feas_U.shape)
            attention = torch.sum(feas_U, dim=1)
            attention = self.gap(attention)#Fgp globalPooling
            # print("1",attention.shape)#128,1,1
            if b ==1:
                attention = self.fc1(attention)
            else:
                attention = self.fc(attention)#Ffc fullyConnect
            # print("2",attention.shape)#32,1,1
            attention = [fc(attention) for fc in self.fcs]
            attention = torch.stack(attention, dim=1)
            # print("3",attention.shape)#32,2,128,1,1
            attention = self.softmax(attention)
            x = torch.sum(feas_U * attention, dim=1)

        # output 
        y = self.conv2(x)
        return y, x # x = Fmid, y = Fout

class SKF(nn.Module):
    def __init__(
        self,student,args, in_channels, out_channels, mid_channel, fuse=False
    ):
        super(SKF, self).__init__()
        self.student = student
        self.Ukd = args.Ukd
        self.args = args
        skfs = nn.ModuleList()

        if fuse == True:
            for idx, in_channel in enumerate(in_channels):
                skfs.append(SK(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
                #in_channels = [16, 32, 64, 128, 256], out_channels = [16, 64, 128, 256, 512]， mid_channel = [64],idx = []
        else:
            for idx, in_channel in enumerate(in_channels):
                skfs.append(SK(in_channel, mid_channel, out_channels[idx], False))
        self.skfs = skfs[::-1]# Reverse（）


    def forward(self, x):
        student_features = self.student(x)
        embeds,student_features,logit = student_features
        if self.Ukd == "Decode":
            if self.args.KDlayers == 4:
                student_features = student_features[3:]
            elif self.args.KDlayers == 3:
                student_features = student_features[4:]
            elif self.args.KDlayers == 2:
                student_features = student_features[5:]
            else:
                print("Ablation Feature Layers Error!")
        elif self.Ukd == "Encode":
            student_features = student_features[:3]
        elif self.Ukd == "EnDe":
            pass
        else:pass
        #student_features = fstudent
        # logit = student_features[1]
        # print(logit.shape)
        # for fea in student_features:
        #     print("stu",fea.shape)
        x = student_features[::-1]#Feature Reverse
        results = []
        out_features, res_features = self.skfs[0](x[0]) #Fout , Fmid
        results.append(out_features)
        
        for features, skf in zip(x[1:], self.skfs[1:]):
            out_features, res_features = skf(features, res_features)
            results.insert(0, out_features) #Reverse again
        return embeds,results,logit

#in_channels,out_channels,all_need_change [16, 32, 64, 128] [512,128,64,32,16]
def build_kd_trans(model,args,in_channels = [256,128,64,32], out_channels = [512,256,128,64],csf = False):
    mid_channel = 128
    print("1.build_fstudent CSF is"+f'_{csf}')
    #embed not be used
    model = SKF(model,args,in_channels, out_channels, mid_channel,fuse=csf)
    return model
