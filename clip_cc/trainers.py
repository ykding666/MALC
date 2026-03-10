from __future__ import print_function, absolute_import

import time
from torch.cuda import amp
from .utils.meters import AverageMeter
import torch

import torch.nn.functional as F
from datetime import datetime


def local_contrastive_loss(local_feats, labels, temperature=0.07):
    """
    local_feats: Tensor, shape = (B, num_regions, D)
    labels: Tensor, shape = (B,)
    """
    B, num_regions, D = local_feats.shape
    # 归一化局部特征
    local_feats = F.normalize(local_feats, dim=-1)

    # 预计算同一身份的 mask (B, B) 对称矩阵；对角线不考虑
    same_id_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    triu_mask = torch.triu(torch.ones_like(same_id_mask, dtype=torch.bool), diagonal=1)
    pos_mask = same_id_mask & triu_mask

    loss_per_region = []
    # 对于局部区域（区域数一般较小）
    for k in range(num_regions):
        # 取第 k 个区域，shape=(B, D)
        feats_k = local_feats[:, k, :]
        # 计算余弦相似度矩阵，shape=(B, B)
        sim_matrix = torch.mm(feats_k, feats_k.t())
        # 取出正样本对，注意这里直接利用 mask 的上三角部分，防止重复计算
        pos_sim = sim_matrix[pos_mask]
        if pos_sim.numel() > 0:
            loss_k = (1 - pos_sim).mean()
        else:
            loss_k = torch.tensor(0.0, device=local_feats.device)
        loss_per_region.append(loss_k)

    loss = sum(loss_per_region) / num_regions
    return loss


class VITFP16(object):
    def __init__(self, encoder, memory=None, lambda_local=0.7):
        super(VITFP16, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.lambda_local = lambda_local

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        scaler = amp.GradScaler()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            with amp.autocast(enabled=True):
                # process inputs
                inputs, labels, indexes = self._parse_data(inputs)
                global_feat, local_feats = self.encoder(inputs)

                loss_global = self.memory(global_feat, labels)

                loss_local = local_contrastive_loss(local_feats, labels, temperature=0.07)
                # 总损失 = 全局损失 + λ_local * 局部损失
                loss = 1.0 * loss_global + self.lambda_local * loss_local

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch + 1, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg))
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes, clothid = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()