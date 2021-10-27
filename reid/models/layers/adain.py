# https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py

import random
import torch
from torch import nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """
    AdaIN
    :param content_feat: (B, C, H, W)
    :param style_feat: (B, C, H, W)
    :return: (B, C, H, W)
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adaptive_instance_normalization_v2(content_feat, style_feat, lam=0.5):
    """

    :param content_feat:
    :param style_feat:
    :param lam: float in [0, 1]. lam = 1, using style feat; lam = 0, using origin feat
    :return:
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    mix_style_std = lam * style_std.expand(size) + (1-lam) * content_std.expand(size)
    mix_style_mean = lam * style_mean.expand(size) + (1-lam) * content_mean.expand(size)

    return normalized_feat * mix_style_std + mix_style_mean


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


class SMMBlock(nn.Module):
    def __init__(self, lam=0.5, rand=False, learnable=False):
        super(SMMBlock, self).__init__()
        self.rand = rand
        self.lam = lam

    def forward(self, x):
        if self.rand:
            lam = random.random()
            lam = lam + (1-self.lam)
            self.lam = lam

        content_feat = x
        batch_indices = torch.randperm(x.shape[0])
        style_feat = x[batch_indices]
        mixed_feat = adaptive_instance_normalization_v2(content_feat, style_feat, self.lam)
        return mixed_feat
