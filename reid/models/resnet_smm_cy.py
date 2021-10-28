from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict

from .layers import CBAM
from .layers.adain import SMMBlock

import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet50_smm_cy']


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_domains=3, lam=0.5, rand_lam=False,
                 learnable_lam=False, smm_stage=-1):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.conv = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool)]))

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.smm_stage = smm_stage
        self.smm_block = SMMBlock(lam, rand=rand_lam, learnable=learnable_lam)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.attention = CBAM(self.num_features)
        # init_model(self.attention)

        self.domain_classifier = nn.Linear(self.num_features, num_domains, bias=False)
        init.normal_(self.domain_classifier.weight, std=0.001)

        self.domain_bn = nn.BatchNorm1d(self.num_features)
        self.domain_bn.bias.requires_grad_(False)
        init.constant_(self.domain_bn.weight, 1)
        init.constant_(self.domain_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, domain_labels=None, output_prob=False):
        bs = x.size(0)
        # x = self.base(x)
        x = self.conv(x)

        if self.smm_stage == 1 and self.training:
            mixed_x, mixed_indices = self.smm_block(x)
            x = torch.cat([x, mixed_x], dim=0)
            mixed_domains = domain_labels[mixed_indices]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        c_mask = self.attention(x)
        content_feat = x * c_mask
        domain_feat = x * (1-c_mask)
        # content_feat = x
        # domain_feat = x

        content_feat = self.gap(content_feat)
        domain_feat = self.gap(domain_feat)
        content_feat = content_feat.view(content_feat.size(0), -1)
        domain_feat = domain_feat.view(domain_feat.size(0), -1)
        domain_feat = self.domain_bn(domain_feat)

        if self.cut_at_pooling:
            return content_feat

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(content_feat))
        else:
            bn_x = self.feat_bn(content_feat)

        if self.training is False and output_prob is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            norm_bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        # if self.num_classes > 0:
        #     prob = self.classifier(bn_x)
        # else:
        #     return bn_x

        prob = self.classifier(bn_x)
        domain_prob = self.domain_classifier(domain_feat)
        if self.norm:
            ori_prob, mixed_prob = torch.chunk(prob, 2, dim=0)
            ori_x, mixed_x = torch.chunk(content_feat, 2, dim=0)
            ori_domain_prob, mixed_domain_prob = torch.chunk(domain_prob, 2, dim=0)
            ori_norm_bn_x, mixed_norm_bn_x = torch.chunk(norm_bn_x, 2, dim=0)
            return [ori_prob, mixed_prob], [ori_x, mixed_x], [ori_domain_prob, mixed_domain_prob], [domain_labels, mixed_domains], [ori_norm_bn_x, mixed_norm_bn_x]
        else:
            ori_prob, mixed_prob = torch.chunk(prob, 2, dim=0)
            ori_x, mixed_x = torch.chunk(content_feat, 2, dim=0)
            ori_domain_prob, mixed_domain_prob = torch.chunk(domain_prob, 2, dim=0)
            return [ori_prob, mixed_prob], [ori_x, mixed_x], [ori_domain_prob, mixed_domain_prob], [domain_labels, mixed_domains]

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


def resnet50_smm_cy(**kwargs):
    return ResNet(50, **kwargs)