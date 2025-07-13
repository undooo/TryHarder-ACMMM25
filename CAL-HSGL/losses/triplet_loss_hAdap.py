from __future__ import absolute_import

import torch
from torch import nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, scaled_hardP, scaled_hardN, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 通过hardP和hardN对dist_mat进行缩放
    # scaled_hardN = scaled_hardN.to(dist_mat.device)
    # scaled_hardP = scaled_hardP.to(dist_mat.device)
    distP_scaled = dist_mat * scaled_hardP
    distN_scaled = dist_mat * scaled_hardN

    # 安全检查 - 正样本
    pos_exists = is_pos.sum() > 0
    if pos_exists:
        dist_ap, relative_p_inds = torch.max(
            distP_scaled[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        dist_ap = dist_ap.squeeze(1)
    else:
        # 没有正样本，使用一个合理的默认值
        dist_ap = torch.zeros(N, device=dist_mat.device)
        relative_p_inds = torch.zeros(N, 1, dtype=torch.long, device=dist_mat.device)
        print(f"警告: 批次中没有找到足够的正样本对!")

    
    # 安全检查 - 负样本
    neg_exists = is_neg.sum() > 0
    if neg_exists:
        dist_an, relative_n_inds = torch.min(
            distN_scaled[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        dist_an = dist_an.squeeze(1)
    else:
        # 没有负样本，使用一个合理的默认值
        dist_an = torch.ones(N, device=dist_mat.device) * 1e6  # 很大的值
        relative_n_inds = torch.zeros(N, 1, dtype=torch.long, device=dist_mat.device)
        print(f"警告: 批次中没有找到足够的负样本对!")
    
   

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletHAdapLoss(nn.Module):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0, cfg=None):
        super(TripletHAdapLoss, self).__init__()
        self.margin = 0.0
        self.hard_factor = hard_factor
        self.cfg = cfg
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat, labels, hardN_mat,hardP_mat, normalize_feature=False):
       
        if self.cfg.LOSS.LABEL_ADJUST_FACTOR == 1.1:
            p_factor = 1.1
            n_factor = 0.9
        elif self.cfg.LOSS.LABEL_ADJUST_FACTOR == 1.2:
            p_factor = 1.2
            n_factor = 0.8
        elif self.cfg.LOSS.LABEL_ADJUST_FACTOR == 1.05:
            p_factor = 1.05
            n_factor = 0.95
        elif self.cfg.LOSS.LABEL_ADJUST_FACTOR == 1.02:
            p_factor = 1.02
            n_factor = 0.98
        else:
            p_factor = 1
            n_factor = 1
          
        #对正样本进行缩放
        scaled_hardP = torch.where(
            hardP_mat > 0.5,
            p_factor,
            torch.ones_like(hardP_mat)
        )
        
        # 对负样本进行缩放
        scaled_hardN = torch.where(
            hardN_mat > 0.5,
            n_factor,
            torch.ones_like(hardN_mat)
        )
             
        noscale_hardN = torch.ones_like(hardN_mat)
        noscale_hardP = torch.ones_like(hardP_mat)
        
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        
        dist_mat = euclidean_dist(global_feat, global_feat) #[128, 128]

        dist_ap, dist_an = hard_example_mining(dist_mat, labels, noscale_hardP, noscale_hardN)
        dist_ap_2, dist_an_2 = hard_example_mining(dist_mat, labels, scaled_hardP, scaled_hardN)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        if self.margin is not None:
            
            loss_1 = self.ranking_loss(dist_an, dist_ap, y)
            loss_2 = self.ranking_loss(dist_an_2, dist_ap_2, y)
        else:
            loss_1 = self.ranking_loss(dist_an - dist_ap, y)
            loss_2 = self.ranking_loss(dist_an_2 - dist_ap_2, y)
            
        if self.cfg.LOSS.LABEL_ADJUST_FACTOR != 0.0:
            loss = loss_1 + 0.5*loss_2
        else:
            loss = loss_1

        return loss