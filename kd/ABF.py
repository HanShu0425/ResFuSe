import torch
from torch import nn
import torch.nn.functional as F

# from .resnet  import *
# from .mobilenet import *

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # upsample residual features
            # y = F.interpolate(y, (shape,shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y, x

class ReviewKD(nn.Module):
    def __init__(
        self, student,Ukd, in_channels, out_channels, mid_channel
    ):
        super(ReviewKD, self).__init__()
        # self.shapes = [1,7,14,28,56]
        self.student = student
        self.Ukd = Ukd
        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, x):
        student_features = self.student(x)
        embeds,student_features,logit = student_features
        if self.Ukd == "Decode":
            student_features = student_features[3:]
        elif self.Ukd == "Encode":
            student_features = student_features[:3]
        elif self.Ukd == "EnDe":
            pass
        else:pass
        x = student_features[::-1]#Feature Reverse
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        # for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
        #     out_features, res_features = abf(features, res_features, shape)
        #     results.insert(0, out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            results.insert(0, out_features) #Reverse again
        return embeds,results, logit


def build_review_kd(model,Ukd,in_channels = [256,128,64,32], out_channels = [512,256,128,64],csf = False):
    
    mid_channel = 128
    print("1.build_fstudent ABF")
    model = ReviewKD(model,Ukd, in_channels, out_channels, mid_channel)
    return model

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all