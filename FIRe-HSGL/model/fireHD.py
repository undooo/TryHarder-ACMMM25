"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

import copy

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class FgClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1, init_center=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.weight = nn.Parameter(copy.deepcopy(init_center))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w)


class AttrAwareLoss(nn.Module):
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, positive_mask):
        inputs = self.scale * inputs
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        log_probs = self.logsoftmax(inputs)
        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_probs).mean(0).sum()
        return loss


class MaxAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)
        return torch.cat((max_f, avg_f), 1)


class HDetectorSimple(nn.Module):
    def __init__(self):
        super(HDetectorSimple, self).__init__()
        
        # 1. 使用线性变换减少维度：从 768 -> 256
        self.feature_reduce = nn.Linear(4096, 256)
        
        # 2. 交互模块：自注意力机制，捕捉两个特征之间的关联
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # 3. 用于硬负样本（hardN_mat）和硬正样本（hardP_mat）的卷积层
        self.conv_hardN = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=1)  # 将512降到32
        self.conv_hardP = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=1)  # 将512降到32
        
        # 4. 用于硬负样本（hardN_mat）的全连接层
        self.fc_hardN = nn.Linear(32, 1)  # 预测输出为1

        # 5. 用于硬正样本（hardP_mat）的全连接层
        self.fc_hardP = nn.Linear(32, 1)  # 预测输出为1
        
        # 激活函数
        self.relu = nn.ReLU()
        # 正则化
        self.dropout = nn.Dropout(0.3)

    def forward(self, feature_i, feature_j):
        # 【A, B】两两配对 
        # 1. 线性变换降维：从 [2016, 768] -> [2016, 256]
        feature_i = self.feature_reduce(feature_i)  # [2016, 256]
        feature_j = self.feature_reduce(feature_j)  # [2016, 256]
        
        # 2. 自注意力交互：将两个特征组合后输入自注意力机制
        combined_features = torch.stack((feature_i, feature_j), dim=1)  # [2016, 2, 256]，拼接后转为 [batch, seq_len, embed_dim]
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        
        # 展平注意力输出为 [2016, 512]
        attn_output = attn_output.reshape(-1, 512)  # 展开为 [2016, 512]
        
        # 3. 卷积降维
        attn_output = attn_output.unsqueeze(-1)  # 调整维度为 [2016, 512, 1]，以适应卷积层
        conv_output_hardN = self.conv_hardN(attn_output)  # [2016, 32, 1]
        conv_output_hardP = self.conv_hardP(attn_output)  # [2016, 32, 1]
        
        # 4. 将卷积输出展平为 [2016, 32]
        conv_output_hardN = conv_output_hardN.reshape(-1, 32)  # [2016, 32]
        conv_output_hardP = conv_output_hardP.reshape(-1, 32)  # [2016, 32]

        # 5. 预测 hardN_mat
        hardN_pred = self.fc_hardN(conv_output_hardN)  # [2016, 32] -> [2016, 1]

        # 6. 预测 hardP_mat
        hardP_pred = self.fc_hardP(conv_output_hardP)  # [2016, 32] -> [2016, 1]

        return hardN_pred, hardP_pred



class FIRe(nn.Module):
    def __init__(self, pool_type='avg', last_stride=1, pretrain=True, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.P_parts = 2
        self.K_times = 1

        resnet = getattr(torchvision.models, 'resnet50')(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        # self.layer4 = resnet.layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        feature_dim = 2048
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'maxavg':
            self.pool = MaxAvgPool2d()
        self.feature_dim = (2 * feature_dim) if pool_type == 'maxavg' else feature_dim

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.FAR_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.FAR_bottleneck.bias.requires_grad_(False)
        self.FAR_bottleneck.apply(weights_init_kaiming)
        self.FAR_classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.FAR_classifier.apply(weights_init_classifier)

    def forward(self, x, fgid=None, is_hard=None, cloth_id=None, label=None, cam_label=None):
        # fgid其实就是cloth_id
        batch_size = x.shape[0]
        x = self.backbone(x)
        global_feat = self.pool(x).flatten(1)  # [B, d]  [B, 4096]
        global_feat_bn = self.bottleneck(global_feat) # [B, 4096]

        if is_hard is not None:
            # 获取样本数
            # 生成所有样本的两两组合，剔除重复的部分 (如 AB 和 BA)
            indices = torch.triu_indices(batch_size, batch_size, offset=1).to("cuda")
            # 初始化 hardN_mat 和 hardP_mat 矩阵
            hardN_mat = torch.zeros(batch_size, batch_size, dtype=torch.int32).to('cuda')
            hardP_mat = torch.zeros(batch_size, batch_size, dtype=torch.int32).to('cuda')
            #self.logger.info("calculating hardN_mat and hardP_mat")

            # 遍历所有两两组合，填充 hardN_mat 和 hardP_mat
            for i, j in zip(indices[0], indices[1]):
                # 如果两个样本都是困难样本，则在 hardN_mat 对应位置为 1
                if is_hard[i] == 1 and is_hard[j] == 1:
                    hardN_mat[i, j] = 1
                    hardN_mat[j, i] = 1  # 对称性
                
                # 如果两个样本是同一人物 (label相同) 且 cam_label 不同，则在 hardP_mat 对应位置为 1
                if label[i] == label[j] and cam_label[i] != cam_label[j]:
                    hardP_mat[i, j] = 1
                    hardP_mat[j, i] = 1  # 对称性
            #self.logger.info("hardN_mat and hardP_mat calculated")
        
        if cloth_id is not None:

            # 生成所有样本的两两组合，剔除重复的部分 (如 AB 和 BA)
            indices = torch.triu_indices(batch_size, batch_size, offset=1).to("cuda")
            # 初始化 hardN_mat 和 hardP_mat 矩阵
            hardN_mat = torch.zeros(batch_size, batch_size, dtype=torch.int32).to('cuda')
            hardP_mat = torch.zeros(batch_size, batch_size, dtype=torch.int32).to('cuda')
            #self.logger.info("calculating hardN_mat and hardP_mat")
            for i, j in zip(indices[0], indices[1]):
                # 如果两个样本是同一人物 (label相同) 且 cam_label 不同，则在 hardP_mat 对应位置为 1
                # if label[i] == label[j] and (cloth_id[i] != cloth_id[j] or cam_label[i] != cam_label[j]):
                if self.cfg.MODEL.HARDFACTOR == 'cloth':
                    hard_factor = cloth_id[i] != cloth_id[j]
                elif self.cfg.MODEL.HARDFACTOR == 'cam':
                    hard_factor = cam_label[i] != cam_label[j]
                elif self.cfg.MODEL.HARDFACTOR == 'cloth+cam':
                    hard_factor = cloth_id[i] != cloth_id[j] or cam_label[i] != cam_label[j]

                if label[i] == label[j] and (hard_factor):
                    hardP_mat[i, j] = 1
                    hardP_mat[j, i] = 1
                
                if label[i] != label[j] and cloth_id[i] == cloth_id[j]:
                    hardN_mat[i, j] = 1
                    hardN_mat[j, i] = 1
            pass
        # 对每对样本拼接它们的 features，并通过 match_branch 预测 hardN 和 hardP
      
        # 初始化预测矩阵
        output_hardN = torch.zeros(batch_size, batch_size).to(global_feat_bn.device) #[bs, bs]
        output_hardP = torch.zeros(batch_size, batch_size).to(global_feat_bn.device) #[bs, bs]

        # 获取所有两两样本的组合索引，批量处理
        indices = torch.triu_indices(batch_size, batch_size, offset=1) #[2, bs*(bs-1)/2]

        # 获取所有拼接后的特征
       
        features1 = global_feat_bn[indices[0]]  # 获取所有 i 对应的 features [bs*(bs-1)/2, 4096]
        features2 = global_feat_bn[indices[1]]  # 获取所有 j 对应的 features [bs*(bs-1)/2, 4096]

        # 批量预测
        hardN_pred, hardP_pred = self.hDetector(features1, features2)

        # 还原到输出矩阵中
        hardN_pred = hardN_pred.to(output_hardN.dtype)
        hardP_pred = hardP_pred.to(output_hardP.dtype)

        output_hardN[indices[0], indices[1]] = hardN_pred.squeeze(1)
        output_hardP[indices[0], indices[1]] = hardP_pred.squeeze(1)
        output_hardN[indices[1], indices[0]] = hardN_pred.squeeze(1)
        output_hardP[indices[1], indices[0]] = hardP_pred.squeeze(1)
        #self.logger.info("hardN_mat and hardP_mat predicted")




        if self.training:
            if fgid is not None:
                part_h = x.shape[2] // self.P_parts
                FAR_parts = []
                for k in range(self.P_parts):
                    part = x[:, :, part_h * k: part_h * (k + 1), :]  # [B, d, h', w]
                    mu = part.mean(dim=[2, 3], keepdim=True)
                    var = part.var(dim=[2, 3], keepdim=True)
                    sig = (var + 1e-6).sqrt()
                    mu, sig = mu.detach(), sig.detach()  # [B, d, 1, 1]
                    id_part = (part - mu) / sig  # [B, d, h, w]

                    neg_mask = fgid.expand(batch_size, batch_size).ne(fgid.expand(batch_size, batch_size).t())  # [B, B]
                    neg_mask = neg_mask.type(torch.float32)
                    sampled_idx = torch.multinomial(neg_mask, num_samples=self.K_times, replacement=False).\
                        transpose(-1, -2).flatten(0)  # [B, K] -> [BK]
                    new_mu = mu[sampled_idx]  # [BK, d, 1, 1]
                    new_sig = sig[sampled_idx]  # [BK, d, 1, 1]

                    id_part = id_part.repeat(self.K_times, 1, 1, 1)
                    FAR_part = (id_part * new_sig) + new_mu  # [B, d, h', w]
                    FAR_parts.append(FAR_part)
                FAR_feat = torch.concat(FAR_parts, dim=2)  # [B, d, h, w]
                FAR_feat = self.pool(FAR_feat).flatten(1)
                FAR_feat_bn = self.FAR_bottleneck(FAR_feat)
                y_FAR = self.FAR_classifier(FAR_feat_bn)
                return global_feat_bn, y_FAR
            
            if is_hard is not None or cloth_id is not None:
                # pretrain 还需返回 hardN_mat 和 hardP_mat，用于计算损失
                return global_feat, [output_hardN, output_hardP, hardN_mat, hardP_mat]
            
            else:
                return global_feat, [output_hardN, output_hardP]
        else:
            if self.args.model_name in ['hdetector']:
                return global_feat, [output_hardN, output_hardP]
            else:
                return global_feat_bn
        
    def load_pretrain_model(self, model_path, num_classes):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state_dict']
        # for item in state_dict.keys():
        #     print(item, '       ------    ', state_dict[item].shape)
        
        new_state_dict = {}
        for item in state_dict.keys():
            if 'classifier' not in item:
                new_state_dict[item] = state_dict[item]
        
        new_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        new_classifier.apply(weights_init_classifier)
        new_state_dict['FAR_classifier.weight'] = new_classifier.weight
        self.load_state_dict(new_state_dict, strict=True)
        # self.FAR_classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        # self.FAR_classifier.apply(weights_init_classifier)
        
        # for item in new_state_dict.keys():
            # print('new_state_dict: ', item, '       ------    ', new_state_dict[item].shape)
        
        print(f"Loaded model from {model_path}")
