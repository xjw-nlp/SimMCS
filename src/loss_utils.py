import torch
import torch.nn as nn


class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss
    

def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # n = score.size()[1]
    dev = score.get_device()
    batch, n = score.shape
    B = 100
    alpha = 0.05
    weights = torch.ones(B, n)
    batch_min_score = torch.min(score, dim=1, keepdim=True)[0]
    batch_max_score = torch.max(score, dim=1, keepdim=True)[0]
    margin_hat = torch.mean(batch_max_score - batch_min_score) / (n - 1)
    batch_margin_bootstrap = torch.empty(batch, B, device=dev)
    # batch_min_score_bootstrap = torch.empty(batch, B, device=dev)
    # batch_max_score_bootstrap = torch.empty(batch, B, device=dev)
    for batch_idx in range(batch):
        score_idx = torch.multinomial(weights, n, replacement=True)
        score_bootstrap = score[batch_idx][score_idx]
        min_score_bootstrap = torch.min(score_bootstrap, dim=1)[0]
        # batch_min_score_bootstrap[batch_idx] = min_score_bootstrap
        max_score_bootstrap = torch.max(score_bootstrap, dim=1)[0]
        # batch_max_score_bootstrap[batch_idx] = max_score_bootstrap
        margin_bootstrap = (max_score_bootstrap - min_score_bootstrap) / (n - 1)
        batch_margin_bootstrap[batch_idx] = margin_bootstrap
    q = torch.tensor([alpha, 1 - alpha], device=dev)
    margin_quantile_point = torch.quantile(batch_margin_bootstrap, q=q, dim=1, keepdim=False).T
    margin_pivot_left = 2 * margin_hat - margin_quantile_point[:, [1]]
    margin_pivot_right = 2 * margin_hat - margin_quantile_point[:, [0]]

    margin = ((margin_pivot_left + margin_pivot_right) / 2).item()
    margin = min(margin, 0.0015)
    margin = max(margin, 0.0005)
    
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            # diff = neg_score - pos_score + margin * i
            # loss = torch.mean(torch.where(diff < 0, min_zero, diff))
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


