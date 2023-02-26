import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

class HCR(nn.Module):
    def __init__(self, weight=1.0):
        super(HCR, self).__init__()
        self.eps = 1e-12
        self.weight = weight

    def pairwise_dist(self, x):
        x_square = x.pow(2).sum(dim=1)
        prod = x @ x.t()
        pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) - 2 * prod).clamp(min=self.eps)
        pdist[range(len(x)), range(len(x))] = 0.
        return pdist
    
    def pairwise_prob(self, pdist):
        return torch.exp(-pdist) 
        
    def hcr_loss(self, h, g):

        q1, q2 = self.pairwise_prob(self.pairwise_dist(h)), self.pairwise_prob(self.pairwise_dist(g))
        return -1 * (q1 * torch.log(q2 + self.eps)).mean() + -1 * ((1 - q1) * torch.log((1 - q2) + self.eps)).mean()

    def forward(self, logits, projections):
        loss_feat = self.hcr_loss(F.normalize(logits, dim=1), F.normalize(projections, dim=1).detach()) * self.weight
        return loss_feat
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        prediction1 = labels.argmax(dim=1)
        temp = (labels!= labels.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
        labels = torch.mul(temp, labels)
        prediction2 = labels.argmax(dim=1)
        temp = (labels != labels.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
        labels = torch.mul(temp, labels)
        prediction3 = labels.argmax(dim=1)
        temp = (labels!= labels.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
        labels = torch.mul(temp, labels)
        prediction4 = labels.argmax(dim=1)
        temp = (labels != labels.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)
        labels = torch.mul(temp, labels)
        prediction5 = labels.argmax(dim=1)
        bsz = prediction1.shape[0]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            prediction1 = prediction1.contiguous().view(-1, 1)
            prediction2 = prediction2.contiguous().view(-1, 1)
            prediction3 = prediction3.contiguous().view(-1, 1)
            prediction4 = prediction4.contiguous().view(-1, 1)
            prediction5 = prediction5.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(prediction1, prediction1.T).float().to(device)
           
            mask2 = torch.eq(prediction2, prediction2.T).float().to(device)
            mask3 = torch.eq(prediction3, prediction3.T).float().to(device)
            mask4 = torch.eq(prediction4, prediction4.T).float().to(device)
            mask5 = torch.eq(prediction5, prediction5.T).float().to(device)
        else:
            mask = mask.float().to(device)
        temp = (mask==mask2)&(mask==mask3)&(mask==mask4)&(mask==mask5)&(mask2==mask3)&(mask2==mask4)&(mask2==mask5)&(mask3==mask4)&(mask3==mask5)&(mask4==mask5)
        # temp = (mask==mask2)&(mask==mask3)&(mask2==mask3)
        
        temp = temp.float()
        mask = torch.where(mask==0,mask,temp)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    


    
class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 3e4

    def __init__(
        self,
        tau=0.5,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)
        
        
        
        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(
            int
        )
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        
        
        return pseudo_label_nxt.cpu(), index


class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, cluster_num=10):
        super().__init__()
        self.cluster_num = cluster_num

    def forward(self, c, pseudo_label):
        pseudo_index = pseudo_label != -1
        pesudo_label = pseudo_label[pseudo_index]
        idx, counts = torch.unique(pseudo_label, return_counts=True)
        freq = pseudo_label.shape[0] / counts.float().to(c.device)
        weight = torch.ones(self.cluster_num).to(c.device)
        weight[idx] = freq
        if pseudo_index.sum() > 0:
            criterion = nn.CrossEntropyLoss(weight=weight).to(c.device)
            loss_ce = criterion(
                c[pseudo_index], pseudo_label[pseudo_index].to(c.device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(c.device)
        return loss_ce