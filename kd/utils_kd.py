
import torch.nn.functional as F
import torch.nn as nn

def U_KnowledgeDistillation(args,model,DEVICE):
    from kd.CSF import build_kd_trans
    from kd.ABF import build_review_kd
    
    if args.Ukd == "Decode":#4
        if args.kd == "response":
            model=model
        elif "csf" in args.kd :
            model = build_kd_trans(model=model,args = args,in_channels=[256,128,64,32],out_channels=[512,256,128,64],csf=True).to(DEVICE)
        elif "abf" in args.kd:
            model = build_review_kd(model=model,Ukd = args.Ukd,in_channels=[256,128,64,32],out_channels=[512,256,128,64],csf=True).to(DEVICE)
        elif "allFeature" in args.kd :
            model = build_kd_trans(model=model,args = args,in_channels=[256,128,64,32],out_channels=[512,256,128,64],csf=False).to(DEVICE)
        # elif args.kd == "mca":
        #     from kd.MCA import build_mca_trans
        #     model = build_mca_trans(model=model,Ukd = args.Ukd,in_channels=[256,128,64,32],out_channels=[512,256,128,64],csf=True).to(DEVICE)
        # elif args.kd == "skembed":
        #     from kd.CSFembed import build_kd_embeds
        #     model = build_kd_embeds(model=model,Ukd = args.Ukd,in_channels=[256,128,64,32],out_channels=[512,256,128,64],csf=True).to(DEVICE)
        elif "resFuSe" in args.kd:
            from kd.ResFuse import build_kd_ResFuSe
            model = build_kd_ResFuSe(model=model,args = args,in_channels=[256,128,64,32],out_channels=[512,256,128,64],module=args.kd).to(DEVICE)
        else:
            print("Not ModelStructure Change")
    elif args.Ukd == "Encode":#3
        if args.kd == "response":
            model=model
        elif args.kd == "csf" :
            model = build_kd_trans(model=model,Ukd = args.Ukd,in_channels=[32,128,256],out_channels=[64,256,512],csf=True).to(DEVICE)
        elif args.kd == "abf":
            model = build_review_kd(model=model,Ukd = args.Ukd,in_channels=[32,128,256],out_channels=[64,256,512],csf=True).to(DEVICE)
        elif args.kd == "allFeature":
            model = build_kd_trans(model=model,Ukd = args.Ukd,in_channels=[32,128,256],out_channels=[64,256,512],csf=False).to(DEVICE)
        elif args.kd == "resFuSe":
            from kd.ResFuse import build_kd_ResFuSe
            model = build_kd_ResFuSe(model=model,Ukd = args.Ukd,in_channels=[32,128,256],out_channels=[64,256,512],csf=True).to(DEVICE)

        else:
            print("Not CSF")
    elif args.Ukd == "EnDe":#3+4
        if args.kd == "response":
            model=model
        elif args.kd == "csf" :
            model = build_kd_trans(model=model,Ukd = args.Ukd,in_channels=[32,128,256,256,128,64,32],out_channels=[64,256,512,512,256,128,64],csf=True).to(DEVICE)
        elif args.kd == "abf":
            model = build_review_kd(model=model,Ukd = args.Ukd,in_channels=[32,128,256,256,128,64,32],out_channels=[64,256,512,512,256,128,64],csf=True).to(DEVICE)
        elif args.kd == "allFeature":
            model = build_kd_trans(model=model,Ukd = args.Ukd,in_channels=[32,128,256,256,128,64,32],out_channels=[64,256,512,512,256,128,64],csf=False).to(DEVICE)
        elif args.kd == "resFuSe":
            from kd.ResFuse import build_kd_ResFuSe
            model = build_kd_ResFuSe(model=model,Ukd = args.Ukd,in_channels=[32,128,256,256,128,64,32],out_channels=[64,256,512,512,256,128,64],csf=True).to(DEVICE)


        else:
            print("Not CSF")
    else:
        print("U-net Knowledge Distillation Error!")
    return model

import torch

# def hcl5(fstudent, fteacher):
#     loss_all = 0.0
#     for fs, ft in zip(fstudent, fteacher):
#         n, c, h, w = fs.shape
#         #loss = l1+l2
#         #Scale自适应
#         #4个尺度
        
#         # loss = F.smooth_l1_loss(fs, ft, reduction='mean')
#         loss = F.mse_loss(fs, ft, reduction='mean')
#         # 动态调整池化尺度
#         # scales = [max(1, h // (2 ** i)) for i in range(3)]
#         # 注意力机制自动学习每个尺度的权重
#         att_weights = []
#         for l in [4,2,1]:
#             if l >=h:
#                 continue
#             fs_pool = F.adaptive_avg_pool2d(fs, (l,l))
#             ft_pool = F.adaptive_avg_pool2d(ft, (l,l))
#             att_weight = torch.mean(torch.cosine_similarity(fs_pool, ft_pool, dim=1))
#             att_weights.append(att_weight)
        
#         # 注意力权重归一化
#         att_weights = torch.softmax(torch.stack(att_weights), dim=0)
#         # print(att_weights.shape)

#         # 注意力权重计算加权损失
#         tot = 0.0
#         for i, scale in enumerate([4,2,1]):
#             fs_pool = F.adaptive_avg_pool2d(fs, (scale, scale))
#             ft_pool = F.adaptive_avg_pool2d(ft, (scale, scale))
#             loss += F.mse_loss(fs, ft, reduction='mean') * att_weights[i]
#             tot += att_weights[i]
#         loss = loss / tot
#         loss_all = loss_all + loss
    
#     return loss_all

# def hcl4(fstudent, fteacher):
#     loss_all = 0.0
#     for fs, ft in zip(fstudent, fteacher):
#         n, c, h, w = fs.shape
#         #loss = l1+l2
#         #Scale自适应
#         #4个尺度
#         # loss=torch.nn.L1Loss()
#         # loss = F.smooth_l1_loss(fs, ft, reduction='mean')
#         loss = F.mse_loss(fs, ft, reduction='mean')
#         # 动态调整池化尺度
#         scales = [max(1, h // (2 ** i)) for i in range(3)]
#         cnt = 1.0
#         tot = 1.0
#         for scale in scales:
#             fs_pool = F.adaptive_avg_pool2d(fs, (scale, scale))
#             ft_pool = F.adaptive_avg_pool2d(ft, (scale, scale))
#             cnt /= 2.0
#             loss += F.mse_loss(fs_pool, ft_pool, reduction='mean')
#             tot += cnt
        
#         loss = loss / tot
#         loss_all = loss_all + loss
    
#     return loss_all

# def hcl3(fstudent, fteacher):
#     loss_all = 0.0
#     # for fs, ft in zip(fstudentNew, fteacher):
#     # loss = F.mae_loss(fs, ft, reduction='mean')
#     # loss = torch.nn.L1Loss(fs, ft, reduction='mean')
#     for fs, ft in zip(fstudent, fteacher):
#         n,c,h,w = fs.shape
#         loss = F.smooth_l1_loss(fs, ft, reduction='mean')
#         cnt = 1.0
#         tot = 1.0
#         for l in [4,2,1]:
#             if l >=h:
#                 continue
#             tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
#             tmpft = F.adaptive_avg_pool2d(ft, (l,l))
#             # print(f'Fstudent:{tmpfs.shape},Fteacher:{tmpft.shape}')
#             cnt /= 2.0
#             loss += F.smooth_l1_loss(fs, ft, reduction='mean') * cnt
#             tot += cnt
#         loss = loss / tot
#         loss_all = loss_all + loss
#     return loss_all

# def hcl2(fstudent, fteacher):
#     loss_all = 0.0
#     for fs, ft in zip(fstudent, fteacher):
#         n, c, h, w = fs.shape
#         #loss = l1+l2
#         #Scale自适应
#         #4个尺度
        
#         # loss = F.smooth_l1_loss(fs, ft, reduction='mean')
#         loss = F.mse_loss(fs, ft, reduction='mean')
#         # 动态调整池化尺度
#         scales = [max(1, h // (2 ** i)) for i in range(3)]
#         # 注意力机制自动学习每个尺度的权重
#         att_weights = []
#         for scale in scales:
#             fs_pool = F.adaptive_avg_pool2d(fs, (scale, scale))
#             ft_pool = F.adaptive_avg_pool2d(ft, (scale, scale))
#             att_weight = torch.mean(torch.cosine_similarity(fs_pool, ft_pool, dim=1))
#             att_weights.append(att_weight)
        
#         # 注意力权重归一化
#         att_weights = torch.softmax(torch.stack(att_weights), dim=0)
#         # print(att_weights.shape)

#         # 注意力权重计算加权损失
#         tot = 0.0
#         for i, scale in enumerate(scales):
#             fs_pool = F.adaptive_avg_pool2d(fs, (scale, scale))
#             ft_pool = F.adaptive_avg_pool2d(ft, (scale, scale))
#             loss += F.mse_loss(fs, ft, reduction='mean') * att_weights[i]
#             tot += att_weights[i]
        
#         loss = loss / tot
#         loss_all = loss_all + loss
    
#     return loss_all

def hcl(fstudent, fteacher):#only Loss not change structures

    loss_all = 0.0
    # for fs, ft in zip(fstudentNew, fteacher):
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
            # print(f'Fstudent:{tmpfs.shape},Fteacher:{tmpft.shape}')
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

def channelKL(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        ft = torch.clamp(ft, 1e-7, 1)
        fs = torch.clamp(fs, 1e-7, 1)
        ft /= torch.sum(ft, dim=(2, 3), keepdim=True)
        fs /= torch.sum(fs, dim=(2, 3), keepdim=True)
        
        # Compute KL divergence channel-wise
        kl_divergence = torch.sum(ft * torch.log(ft / (fs + 1e-7)), dim=(2, 3))
        
        # Take the mean over channels
        loss_all += torch.mean(kl_divergence)
    return loss_all

def spatialKL(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        ft = torch.clamp(ft, 1e-7, 1)
        fs = torch.clamp(fs, 1e-7, 1)
        ft /= torch.sum(ft, dim=(2, 3), keepdim=True)
        fs /= torch.sum(fs, dim=(2, 3), keepdim=True)
        
        # Reshape y_true and y_pred to be 2D matrices (spatial locations)
        batch_size, _, height, width = ft.size()
        ft_2d = ft.reshape(batch_size, -1)
        fs_2d = fs.reshape(batch_size, -1)
        
        # Compute KL divergence spatial-wise
        kl_divergence = torch.sum(ft_2d * torch.log(ft_2d / (fs_2d + 1e-7)), dim=1)
        loss_all += torch.mean(kl_divergence)
    
    return loss_all

def mse(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        loss = F.mse_loss(fs, ft, reduction='mean')
        loss_all = loss_all + loss
    return loss_all

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_vectors, student_vectors):
        # teacher_vectors: (batch, feature_dim)
        # student_vectors: (batch, feature_dim)
        batch_size,num_classes, _, _= teacher_vectors.shape

        # 归一化特征向量
        teacher_features = nn.functional.normalize(teacher_vectors, dim=1)
        student_features = nn.functional.normalize(student_vectors, dim=1)

        loss = 0.0
        for i in range(batch_size):
            # 提取当前样本的学生特征和教师特征
            student_sample = student_features[i]  # (9, 224, 224)
            teacher_sample = teacher_features[i]  # (9, 224, 224)

            # 将特征展平为 (num_classes, height * width)
            student_sample = nn.functional.normalize(student_sample.view(num_classes, -1), dim=1)  # (9, 224*224)
            teacher_sample = nn.functional.normalize(teacher_sample.view(num_classes, -1), dim=1)  # (9, 224*224)
            # 计算相似度矩阵 (num_classes, num_classes)
            similarity_matrix = torch.matmul(student_sample, teacher_sample.t())  # (9, 9)

            # 计算正样本对和负样本对的相似度
            positive_pairs = torch.diag(similarity_matrix)  # 正样本对的相似度 (9,)
            negative_pairs = similarity_matrix.flatten()  # 负样本对的相似度 (9*9,)

            # InfoNCE 损失
            numerator = torch.exp(positive_pairs / self.temperature)
            denominator = torch.sum(torch.exp(negative_pairs / self.temperature))
            batch_loss = -torch.log(numerator / denominator)

            # 累加当前样本的损失
            loss += batch_loss.mean()
        # 平均损失
        loss /= batch_size
        # # 计算相似度矩阵
        # similarity_matrix = torch.matmul(teacher_vectors, student_vectors.T) / self.temperature

        # # 对角线上的元素是正样本对
        # positive_pairs = torch.diag(similarity_matrix)

        # # 负样本对是矩阵中非对角线的元素
        # negative_pairs = similarity_matrix[~torch.eye(batch_size).bool()].view(batch_size, -1)

        # # 计算损失
        # loss = -torch.log(torch.exp(positive_pairs) / torch.sum(torch.exp(negative_pairs), dim=1))
        return loss.mean()

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_channels, output_dim):
#         super(FeatureExtractor, self).__init__()
#         self.conv = nn.Conv2d(input_channels, output_dim, kernel_size=1)  # 1x1 卷积
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.flatten(x)
#         return x

def contrastiveKD(stu_out,tea_out):
    DEVICE = torch.device("cuda")
    stu_out = stu_out.to(DEVICE)
    tea_out = tea_out.to(DEVICE)
    b,num_classes,h,w = stu_out.shape
    # feature_dim = 128  # 特征向量的维度
    # fea_ext = FeatureExtractor(num_classes, feature_dim).to(DEVICE)
    # tea_vec = fea_ext(tea_out)  # (batch, feature_dim)
    # stu_vec = fea_ext(stu_out)  # (batch, feature_dim)

    contrastive_loss = ContrastiveLoss(temperature=2).to(DEVICE)

    loss_contrastive = contrastive_loss(tea_out, stu_out).to(torch.device("cuda:1"))
    

    return loss_contrastive


def KDloss(args,loss_logits,f,fT,x,xT):
    if args.loss == "hcl":
        lossFea = hcl
    # elif args.loss == "hcl2":
    #     lossFea = hcl2
    # elif args.loss == "hcl3":
    #     lossFea = hcl3
    # elif args.loss == "hcl4":
    #     lossFea = hcl4
    # elif args.loss == "hcl5":
    #     lossFea = hcl5
    elif args.loss == "channelKL":
        lossFea = channelKL
    elif args.loss == "spatialKL":
        lossFea = spatialKL
    elif args.loss == "mse":
        lossFea = mse
    else:
        print("You not choice Your KDloss!")
        lossFea = hcl
    
    
    if "Contrastive" in args.kd:
        loss_feature = lossFea(f,fT)
        loss_contrastive = contrastiveKD(x,xT)# 1 / 0.3 / 
        loss = loss_feature*1 + loss_contrastive*0.2 + loss_logits
    elif "resFuSe" in args.kd:
        loss_feature = lossFea(f,fT) #need temp and alpha,hcl is low
        loss = loss_feature*1 + loss_logits
    # elif args.kd == "resFuSe+Contrastive":#added by han on 24_12_22
    #     loss_feature = lossFea(f,fT)
    #     loss_contrastive = contrastiveKD(x,xT)# 1 / 0.3 / 
    #     loss = loss_feature*1 + loss_contrastive*0.2 + loss_logits
    # elif args.kd == "Contrastive":
    #     loss_contrastive = contrastiveKD(x,xT)
    #     loss =  loss_contrastive*1 + loss_logits
    #     loss_feature = loss_contrastive
    elif args.kd == "csf":
        loss_feature = lossFea(f,fT) #need temp and alpha,hcl is low
        loss = loss_feature*1 + loss_logits
    # elif args.kd == "csf+Contrastive":
    #     loss_feature = lossFea(f,fT) #need temp and alpha,hcl is low
    #     loss_contrastive = contrastiveKD(x,xT)# 1 / 0.3 / 0.2
    #     loss = loss_feature*1 + loss_contrastive*0.2 + loss_logits
    #     # print(f"Loss_Logit:{loss_logits},Loss_Features:{loss_feature}")
    elif args.kd == "abf":
        loss_feature = lossFea(f,fT) #need temp and alpha,hcl is low
        loss = loss_feature*1 + loss_logits
        # print(f"Loss_Logit:{loss_logits},Loss_Features:{loss_feature}")
    elif args.kd == "allFeature":
        loss_feature = lossFea(f,fT)
        loss = loss_feature + loss_logits
        # print(f"Loss_Logit:{loss_logits},Loss_Features:{loss_feature}")
    elif args.kd == "response":
        from kd.ConvF import loss_Softmax
        alpha = 0.3
        loss_feature = loss_Softmax(x,xT)
        # print("softmax:",loss_feature)
        # print("logit:",loss_logits)
        loss = loss_feature*alpha + loss_logits*(1-alpha)
    elif args.kd == "logit":#no_Knowledge_Dstillation
        loss = loss_logits
        loss_feature = 0
        print("logit noFeatureLoss") 
    else:
        print("No Loss")

    if args.addResponse == "yes":
        from kd.ConvF import loss_Softmax
        alpha = 0.3
        loss_response = loss_Softmax(x,xT)
        loss = loss_feature + loss_logits*(1-alpha) + loss_response*alpha
    return loss_feature, loss