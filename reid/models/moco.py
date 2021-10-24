from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'moco_resnet50']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
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

        if not pretrained:
            self.reset_params()

    def forward(self, x, output_prob=False):
        bs = x.size(0)
        # x = self.base(x)
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False and output_prob is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            norm_bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        if self.norm:
            return prob, x, norm_bn_x
        else:
            return prob, x

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


class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, dim=2048, m=0.8, K=4096, T=0.07):
        super(MoCo, self).__init__()
        # dimension of extracted features and the size of queue
        self.dim = dim
        self.K = K
        self.T = T
        self.m = m

        # encoder_q: base feature extractor
        # encoder_k: auxiliary feature extractor
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # initialize parameters and freeze encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Method 1：updating queue with new features of new examples
        # Method 2: fixed queue storing the mean features of all ids
        self.register_buffer('queue', torch.randn(self.dim, self.K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        new_state_dict = OrderedDict()
        model_dict = self.encoder_k.state_dict()
        for param_q_key, param_q in self.encoder_q.named_parameters():
            if param_q_key in model_dict and model_dict[param_q_key].size() == param_q.size():
                new_state_dict[param_q_key] = model_dict[param_q_key] * (1-self.m) + param_q * self.m
        model_dict.update(new_state_dict)
        self.encoder_k.load_state_dict(model_dict)
        # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def initialize_queue(self, dataloader):
        while True:
            source_inputs = dataloader.next()
            im_k = source_inputs['im_k'].cuda()
            _, k = self.encoder_k(im_k)
            last_ptr = int(self.queue_ptr)
            self._dequeue_and_enqueue(k)
            curr_ptr = int(self.queue_ptr)
            if curr_ptr <= last_ptr:
                break
        # normalize features in queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

    def forward(self, inputs):

        if not self.training:
            feat = self.encoder_q(inputs)
            return feat

        im_q = inputs['im_q']
        im_k = inputs['im_k']

        pred_q, feat_q = self.encoder_q(im_q)
        # feat_q = F.normalize(feat_q)
        # FIXME: normalize feat_q

        with torch.no_grad():
            self._momentum_update_key_encoder()
            pred_k, feat_k = self.encoder_k(im_k)
            # feat_q = F.normalize(feat_q)
            # FIXME: normalize feat_k

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [feat_q, self.queue.clone().detach()])

        # logits: Nx(K+1)
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(feat_k)

        return pred_q, feat_q, logits, labels


class MoCo2(nn.Module):
    def __init__(self, encoder_q, encoder_k, dim=2048, m=0.8, K=4096, T=0.07):
        super(MoCo2, self).__init__()
        # dimension of extracted features and the size of queue
        self.dim = dim
        self.K = K
        self.T = T
        self.m = m

        # encoder_q: base feature extractor
        # encoder_k: auxiliary feature extractor
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # initialize parameters and freeze encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Method 1：updating queue with new features of new examples
        # Method 2: fixed queue storing the mean features of all ids
        self.register_buffer('queue', torch.randn(self.dim, self.K))
        self.register_buffer('gt_queue', torch.randn(self.K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        new_state_dict = OrderedDict()
        model_dict = self.encoder_k.state_dict()
        for param_q_key, param_q in self.encoder_q.named_parameters():
            if param_q_key in model_dict and model_dict[param_q_key].size() == param_q.size():
                new_state_dict[param_q_key] = model_dict[param_q_key] * (1-self.m) + param_q * self.m
        model_dict.update(new_state_dict)
        self.encoder_k.load_state_dict(model_dict)
        # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.gt_queue[:, ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def initialize_queue(self, dataloader):
        while True:
            source_inputs = dataloader.next()
            im_k = source_inputs['im_k'].cuda()
            pids = source_inputs['pids'].cuda()
            _, k = self.encoder_k(im_k)
            last_ptr = int(self.queue_ptr)
            self._dequeue_and_enqueue(k)
            curr_ptr = int(self.queue_ptr)
            if curr_ptr <= last_ptr:
                break
        # normalize features in queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

    def forward(self, inputs):

        if not self.training:
            feat = self.encoder_q(inputs)
            return feat

        im_q = inputs['im_q']
        im_k = inputs['im_k']

        pred_q, feat_q = self.encoder_q(im_q)
        # feat_q = F.normalize(feat_q)
        # FIXME: normalize feat_q

        with torch.no_grad():
            self._momentum_update_key_encoder()
            pred_k, feat_k = self.encoder_k(im_k)
            # feat_q = F.normalize(feat_q)
            # FIXME: normalize feat_k

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [feat_q, feat_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [feat_q, self.queue.clone().detach()])

        # logits: Nx(K+1)
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(feat_k)

        return pred_q, feat_q, logits, labels


def moco_resnet50(dim=2048, m=0.9, K=1024, T=0.07, **kwargs):
    encoder_q = resnet50(**kwargs)
    encoder_k = resnet50(**kwargs)
    moco = MoCo(encoder_q=encoder_q,
                encoder_k=encoder_k,
                dim=dim,
                m=m,
                K=K,
                T=T)
    return moco







