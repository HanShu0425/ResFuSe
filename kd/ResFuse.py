import torch
from torch import nn
import torch.nn.functional as F

class CustomConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(CustomConvModule, self).__init__()      
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)   
        if norm_cfg is not None:
            if norm_cfg['type'] == 'GN':
                num_groups = norm_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups, out_channels, eps=1e-5)
        else:
            self.norm = None   
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        
        return x

# class rSoftMax(nn.Module):
#     def __init__(self, radix, cardinality):
#         super().__init__()
#         self.radix = radix
#         self.cardinality = cardinality

#     def forward(self, x):
#         batch = x.size(0)
#         if self.radix > 1:
#             x = x.view(batch, 1, 2, -1).transpose(1, 2)
#             x = F.softmax(x, dim=1)
#             x = x.reshape(batch, -1)
#         else:
#             x = torch.sigmoid(x)
#         return x
    


class ResFS(nn.Module):#!!!!!
    def __init__(self, in_channel,dim,out_channel,fuse, M=2, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ResFS, self).__init__()
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.squeeze = CustomConvModule(
                sum((dim,dim)),
                dim,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')
            )
            self.dim = dim
            self.channel = dim // M  
            assert dim == self.channel * M
            self.d = self.channel // r  
            self.M = M
            self.proj = nn.Linear(dim,dim) 

            self.act_layer = act_layer()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(dim,self.d) 
            self.fc2 = nn.Linear(self.d, self.M*self.channel)
            
            self.softmax = nn.Softmax(dim=1)
            self.proj_head = nn.Linear(self.channel, dim)
            self.gate = nn.Sequential(
                nn.Conv2d(2*dim, dim, kernel_size=1),nn.Sigmoid())
            
    def forward(self, input_feats,y=None,shape=None):#Ours Methods
        input_feats = self.conv1(input_feats)

        if self.fuse:
            shape = input_feats.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")

            input_feats = torch.cat([input_feats,y],dim=1)
            input_feats = self.squeeze(input_feats)# bs,dim,H,W 128,56,56
            bs, channel, H, W = input_feats.shape
            input_groups = input_feats.reshape(bs,self.M, self.channel, H, W)
            #bs 2 channel H W 56 56
            feats = self.proj(input_feats.view(bs,H*W,channel)) #[bs,H*W,dim]
            #FullyConnect 128,56,56
            feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ActivationLayer 
            feats = self.act_layer(feats)#Fa #bs,h*w,dim
            feats = feats.permute(0,2,1).reshape(bs,self.dim,H,W) 
            #GlobalPooling 1*1*128
            feats_S = self.gap(feats) #[bs,dim,1,1] Fgp
            #FullyConnect 32
            feats_Z = self.fc1(feats_S.squeeze()) #[bs,d] Ffc #bs,d=32?? 128
            #ActivationLayer 32
            feats_Z = self.act_layer(feats_Z) #[bs,d] Fa

            attention_vectors = self.fc2(feats_Z) #[bs,M*channel=dim]#32,128
            attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)#32,2,64,1,1

            #softmax
            attention_vectors = self.softmax(attention_vectors)#32,2,64
            feats_V = torch.sum(input_groups * attention_vectors, dim=1) #[bs,channel,H,W]#32,64,14,14
            #FullyConnect 128,64*64 Linear-->Channel
            feats_V = self.proj_head(feats_V.reshape(bs,self.channel,H*W).permute(0,2,1)) #[bs,H*W,dim]#32,196,128
            #FullyConnext 128,64,64
            feats_V = feats_V.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ResNeSt: Y = f(x) + x
            #ResFuSe: Y = f(x)*(1-k) + x*k
            # input_feats = feats_proj + feats_V #[bs,dim,H,W]
            gate = self.gate(torch.cat((feats_proj, feats_V), dim=1))
            input_feats = feats_proj * gate + feats_V * (1 - gate)
        y = self.conv2(input_feats)
        return y,input_feats     

class ResFS_Res(nn.Module):#!!!!!
    def __init__(self, in_channel,dim,out_channel,fuse, M=2, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ResFS_Res, self).__init__()
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.squeeze = CustomConvModule(
                sum((dim,dim)),
                dim,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')
            )
            self.dim = dim
            self.channel = dim // M  
            assert dim == self.channel * M
            self.d = self.channel // r  
            self.M = M
            self.proj = nn.Linear(dim,dim) 

            self.act_layer = act_layer()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(dim,self.d) 
            self.fc2 = nn.Linear(self.d, self.M*self.channel)
            
            self.softmax = nn.Softmax(dim=1)
            self.proj_head = nn.Linear(self.channel, dim)
            # self.gate = nn.Sequential(
            #     nn.Conv2d(2*dim, dim, kernel_size=1),nn.Sigmoid())
            
    def forward(self, input_feats,y=None,shape=None):#Ours Methods
        input_feats = self.conv1(input_feats)

        if self.fuse:
            shape = input_feats.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")

            input_feats = torch.cat([input_feats,y],dim=1)
            input_feats = self.squeeze(input_feats)# bs,dim,H,W 128,56,56
            bs, channel, H, W = input_feats.shape
            input_groups = input_feats.reshape(bs,self.M, self.channel, H, W)
            #bs 2 channel H W 56 56
            feats = self.proj(input_feats.view(bs,H*W,channel)) #[bs,H*W,dim]
            #FullyConnect 128,56,56
            feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ActivationLayer 
            feats = self.act_layer(feats)#Fa #bs,h*w,dim
            feats = feats.permute(0,2,1).reshape(bs,self.dim,H,W) 
            #GlobalPooling 1*1*128
            feats_S = self.gap(feats) #[bs,dim,1,1] Fgp
            #FullyConnect 32
            feats_Z = self.fc1(feats_S.squeeze()) #[bs,d] Ffc #bs,d=32?? 128
            #ActivationLayer 32
            feats_Z = self.act_layer(feats_Z) #[bs,d] Fa

            attention_vectors = self.fc2(feats_Z) #[bs,M*channel=dim]#32,128
            attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)#32,2,64,1,1

            #softmax
            attention_vectors = self.softmax(attention_vectors)#32,2,64
            feats_V = torch.sum(input_groups * attention_vectors, dim=1) #[bs,channel,H,W]#32,64,14,14
            #FullyConnect 128,64*64 Linear-->Channel
            feats_V = self.proj_head(feats_V.reshape(bs,self.channel,H*W).permute(0,2,1)) #[bs,H*W,dim]#32,196,128
            #FullyConnext 128,64,64
            feats_V = feats_V.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ResNeSt: Y = f(x) + x
            #ResFuSe: Y = f(x)*(1-k) + x*k
            input_feats = feats_proj + feats_V #[bs,dim,H,W]
            # gate = self.gate(torch.cat((feats_proj, feats_V), dim=1))
            # input_feats = feats_proj * gate + feats_V * (1 - gate)
        y = self.conv2(input_feats)
        return y,input_feats
    
class ResFS_noRes(nn.Module):#!!!!!
    def __init__(self, in_channel,dim,out_channel,fuse, M=2, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ResFS_noRes, self).__init__()
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.squeeze = CustomConvModule(
                sum((dim,dim)),
                dim,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')
            )
            self.dim = dim
            self.channel = dim // M  
            assert dim == self.channel * M
            self.d = self.channel // r  
            self.M = M
            self.proj = nn.Linear(dim,dim) 

            self.act_layer = act_layer()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(dim,self.d) 
            self.fc2 = nn.Linear(self.d, self.M*self.channel)
            
            self.softmax = nn.Softmax(dim=1)
            self.proj_head = nn.Linear(self.channel, dim)
            # self.gate = nn.Sequential(
            #     nn.Conv2d(2*dim, dim, kernel_size=1),nn.Sigmoid())
            
    def forward(self, input_feats,y=None,shape=None):#Ours Methods
        input_feats = self.conv1(input_feats)

        if self.fuse:
            shape = input_feats.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")

            input_feats = torch.cat([input_feats,y],dim=1)
            input_feats = self.squeeze(input_feats)# bs,dim,H,W 128,56,56
            bs, channel, H, W = input_feats.shape
            input_groups = input_feats.reshape(bs,self.M, self.channel, H, W)
            #bs 2 channel H W 56 56
            feats = self.proj(input_feats.view(bs,H*W,channel)) #[bs,H*W,dim]
            #FullyConnect 128,56,56
            feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ActivationLayer 
            feats = self.act_layer(feats)#Fa #bs,h*w,dim
            feats = feats.permute(0,2,1).reshape(bs,self.dim,H,W) 
            #GlobalPooling 1*1*128
            feats_S = self.gap(feats) #[bs,dim,1,1] Fgp
            #FullyConnect 32
            feats_Z = self.fc1(feats_S.squeeze()) #[bs,d] Ffc #bs,d=32?? 128
            #ActivationLayer 32
            feats_Z = self.act_layer(feats_Z) #[bs,d] Fa

            attention_vectors = self.fc2(feats_Z) #[bs,M*channel=dim]#32,128
            attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)#32,2,64,1,1

            #softmax
            attention_vectors = self.softmax(attention_vectors)#32,2,64
            feats_V = torch.sum(input_groups * attention_vectors, dim=1) #[bs,channel,H,W]#32,64,14,14
            #FullyConnect 128,64*64 Linear-->Channel
            feats_V = self.proj_head(feats_V.reshape(bs,self.channel,H*W).permute(0,2,1)) #[bs,H*W,dim]#32,196,128
            #FullyConnext 128,64,64
            feats_V = feats_V.permute(0,2,1).reshape(bs,self.dim,H,W)
            #ResNeSt: Y = f(x) + x
            #ResFuSe: Y = f(x)*(1-k) + x*k
            input_feats = feats_V #[bs,dim,H,W]
            # gate = self.gate(torch.cat((feats_proj, feats_V), dim=1))
            # input_feats = feats_proj * gate + feats_V * (1 - gate)
        y = self.conv2(input_feats)
        return y,input_feats

class ResFS_onlyCat(nn.Module):#!!!!!
    def __init__(self, in_channel,dim,out_channel,fuse, M=2, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ResFS_onlyCat, self).__init__()
        self.fuse = fuse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.squeeze = CustomConvModule(
                sum((dim,dim)),
                dim,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')
            )
            # self.dim = dim
            # self.channel = dim // M  
            # assert dim == self.channel * M
            # self.d = self.channel // r  
            # self.M = M
            # self.proj = nn.Linear(dim,dim) 

            # self.act_layer = act_layer()
            # self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # self.fc1 = nn.Linear(dim,self.d) 
            # self.fc2 = nn.Linear(self.d, self.M*self.channel)
            
            # self.softmax = nn.Softmax(dim=1)
            # self.proj_head = nn.Linear(self.channel, dim)
            # self.gate = nn.Sequential(
            #     nn.Conv2d(2*dim, dim, kernel_size=1),nn.Sigmoid())
            
    def forward(self, input_feats,y=None,shape=None):#Ours Methods
        input_feats = self.conv1(input_feats)

        if self.fuse:
            shape = input_feats.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")

            input_feats = torch.cat([input_feats,y],dim=1)
            input_feats = self.squeeze(input_feats)# bs,dim,H,W 128,56,56
        y = self.conv2(input_feats)
        return y,input_feats
    
# class ResFS(nn.Module):#!!!!!
#     def __init__(self, in_channel,dim,out_channel,fuse, M=2, r=2, act_layer=nn.GELU):
#         """ Constructor
#         Args:
#             dim: input channel dimensionality.
#             M: the number of branchs.
#             r: the ratio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(ResFS, self).__init__()
#         self.fuse = fuse
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(dim, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
#             nn.BatchNorm2d(out_channel),
#         )
#         if fuse:
#             self.squeeze = CustomConvModule(
#                 sum((dim,dim)),
#                 dim,
#                 1,
#                 conv_cfg=None,
#                 norm_cfg=dict(type='GN', num_groups=32),
#                 act_cfg=dict(type='ReLU')
#             )
#             self.dim = dim
#             self.channel = dim // M  
#             assert dim == self.channel * M
#             self.d = self.channel // r  
#             self.M = M
#             self.proj = nn.Linear(dim,dim) 

#             self.act_layer = act_layer()
#             self.gap = nn.AdaptiveAvgPool2d((1, 1))
#             self.fc1 = nn.Linear(dim,self.d) 
#             self.fc2 = nn.Linear(self.d, self.M*self.channel)
            
#             self.softmax = nn.Softmax(dim=1)
#             self.proj_head = nn.Linear(self.channel, dim)
#             self.gate = nn.Sequential(
#                 nn.Conv2d(dim, dim, kernel_size=1),nn.Sigmoid())
            
#     def forward(self, input_feats,y=None,shape=None):#Ours Methods
#         input_feats = self.conv1(input_feats)

#         if self.fuse:
#             shape = input_feats.shape[-2:]
#             y = F.interpolate(y, shape, mode="nearest")

#             input_feats = torch.cat([input_feats,y],dim=1)
#             input_feats = self.squeeze(input_feats)# bs,dim,H,W 128,56,56
#             bs, channel, H, W = input_feats.shape
#             input_groups = input_feats.reshape(bs,self.M, self.channel, H, W)
#             #bs 2 channel H W 56 56
#             feats = self.proj(input_feats.view(bs,H*W,channel)) #[bs,H*W,dim]
            
#             #FullyConnect 128,56,56
#             feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,H,W)

#             #ActivationLayer 
#             feats = self.act_layer(feats)#Fa #bs,h*w,dim
            
#             feats = feats.permute(0,2,1).reshape(bs,self.dim,H,W) 

#             #GlobalPooling 1*1*128
#             feats_S = self.gap(feats) #[bs,dim,1,1] Fgp

#             #FullyConnect 32
#             feats_Z = self.fc1(feats_S.squeeze()) #[bs,d] Ffc #bs,d=32?? 128

#             #ActivationLayer 32
#             feats_Z = self.act_layer(feats_Z) #[bs,d] Fa


#             attention_vectors = self.fc2(feats_Z) #[bs,M*channel]#32,128
#             attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1, 1)#32,2,64,1,1

#             #softmax
#             attention_vectors = self.softmax(attention_vectors)#32,2,64
#             feats_V = torch.sum(input_groups * attention_vectors, dim=1) #[bs,channel,H,W]#32,64,14,14
            
#             #FullyConnect 128,64*64 Linear-->Channel
#             feats_V = self.proj_head(feats_V.reshape(bs,self.channel,H*W).permute(0,2,1)) #[bs,H*W,dim]#32,196,128
            
#             #FullyConnext 128,64,64
#             feats_V = feats_V.permute(0,2,1).reshape(bs,self.dim,H,W)

#             #ResNeSt: Y = f(x) + x
#             #ResFuSe: Y = f(x)*(1-k) + x*k
#             # input_feats = feats_proj + feats_V #[bs,dim,H,W]
#             gate = self.gate(feats_proj)
#             input_feats = feats_proj * gate + feats_V * (1 - gate)
#         y = self.conv2(input_feats)

#         return y,input_feats

class SKF(nn.Module):
    def __init__(
        self,student,args, in_channels, out_channels, mid_channel, fuse="ResFuSe"
    ):
        super(SKF, self).__init__()
        self.student = student
        self.Ukd = args.Ukd
        self.args = args
        skfs = nn.ModuleList()

        if fuse == "resFuSe":
            print("=============resFuSe================")
            for idx, in_channel in enumerate(in_channels):
                skfs.append(ResFS(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
                #in_channels = [16, 32, 64, 128, 256], out_channels = [16, 64, 128, 256, 512]， mid_channel = [64],idx = []
        elif fuse == "resFuSe_Res":
            print("=============resFuSe_Res================")
            for idx, in_channel in enumerate(in_channels):
                skfs.append(ResFS_Res(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        elif fuse == "resFuSe_noRes":
            print("=============resFuSe_noRes================")
            for idx, in_channel in enumerate(in_channels):
                skfs.append(ResFS_noRes(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        elif fuse == "resFuSe_onlyCat":
            print("=============resFuSe_onlyCat================")
            for idx, in_channel in enumerate(in_channels):
                skfs.append(ResFS_onlyCat(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        else:
            print("=============allFeature================")
            for idx, in_channel in enumerate(in_channels):
                skfs.append(ResFS(in_channel, mid_channel, out_channels[idx], False))
        self.skfs = skfs[::-1]# Reverse（）


    def forward(self, x):
        student_features = self.student(x)
        embeds,student_features,logit = student_features
        if self.Ukd == "Decode":
            student_features = student_features[7-self.args.KDlayers:]
            # if self.args.KDlayers == 4:
            #     student_features = student_features[3:]
            # elif self.args.KDlayers == 3:
            #     student_features = student_features[4:]
            # elif self.args.KDlayers == 2:
            #     student_features = student_features[5:]
            # else:
            #     print("Ablation Feature Layers Error!")
        elif self.Ukd == "Encode":
            student_features = student_features[:3]
        elif self.Ukd == "EnDe":
            student_features = student_features[7-self.args.KDlayers:]
            # if self.args.KDlayers == 7:
            #     student_features = student_features
            # elif self.args.KDlayers == 6:
            #     student_features = student_features[1:]
            # elif self.args.KDlayers == 5:
            #     student_features = student_features[2:]
            # elif self.args.KDlayers == 4:
            #     student_features = student_features[3:]
            # elif self.args.KDlayers == 3:
            #     student_features = student_features[4:]
            # elif self.args.KDlayers == 2:
            #     student_features = student_features[5:]
            # else:
            #     print("Ablation Feature Layers Error!")
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
def build_kd_ResFuSe(model,args,in_channels = [256,128,64,32], out_channels = [512,256,128,64],module = "ResFuSe"):
    mid_channel = 128
    print("1.build_fstudent ResFuSe")
    #embed not be used
    model = SKF(model,args,in_channels, out_channels, mid_channel,fuse=module)
    return model

