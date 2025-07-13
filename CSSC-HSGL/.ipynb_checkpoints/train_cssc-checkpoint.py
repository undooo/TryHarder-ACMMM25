"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ICASSP 2025 paper: Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification
URL: arxiv.org/abs/2405.16597
GitHub: https://github.com/QizaoWang/CSSC-CCReID
"""

from tqdm import tqdm
from utils.util import AverageMeter
import torch

def train(args, epoch, train_loader, model, optimizer, scheduler, class_criterion, metric_criterion, use_gpu):
    tri_start_epoch = args.tri_start_epoch
    id_losses = AverageMeter()
    tri_losses = AverageMeter()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if args.dataset in ['prcc', 'ltcc']:
            img, pid, fgid, camid = data
        else:
            img, pid, _ = data

        if use_gpu:
            img, pid, fgid, camid = img.cuda(), pid.cuda(), fgid.cuda(), camid.cuda()
        
        bs = img.size(0)
        indices = torch.triu_indices(bs, bs, offset=1).to("cuda")
        hardN_mat = torch.zeros(bs, bs, dtype=torch.int32).to('cuda')
        hardP_mat = torch.zeros(bs, bs, dtype=torch.int32).to('cuda')
        # 遍历所有两两组合，填充 hardN_mat 和 hardP_mat
        for i, j in zip(indices[0], indices[1]):
            # 若两样本id不同，但衣服id相同，则在 hardN_mat 对应位置为 1
            if pid[i] != pid[j] and fgid[i] == fgid[j]:
                hardN_mat[i, j] = 1
                hardN_mat[j, i] = 1
            
            # 若两样本id相同，但衣服id不同，则在 hardP_mat 对应位置为 1
            if pid[i] == pid[j] and (fgid[i] != fgid[j] or camid[i] != camid[j]):
                hardP_mat[i, j] = 1
                hardP_mat[j, i] = 1
        
        model.train()
        feat_list, y_list = model(img)

        loss = 0
        for y in y_list:
            id_loss = class_criterion(y, pid)
            loss += id_loss

        if epoch > tri_start_epoch:
            for feat in feat_list:
                tri_loss = metric_criterion(feat, pid, hardN_mat, hardP_mat)
                loss += args.trip_weight * tri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        id_losses.update(id_loss.item(), pid.size(0))
        if epoch > tri_start_epoch:
            tri_losses.update(tri_loss.item(), pid.size(0))

    
    print('Ep{0} Id:{id_loss.avg:.4f} Tri:{tri_loss.avg:.4f} '.format(
        epoch, id_loss=id_losses, tri_loss=tri_losses))

    scheduler.step()
