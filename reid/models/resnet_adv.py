from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

from collections import OrderedDict

from .layers.attention import CBAM

from ..loss import CrossEntropyLabelSmooth, TripletLoss


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50_adv', 'resnet101', 'resnet152']


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, n_inputs // 2)
        self.dropout = nn.Dropout(0.5)
        self.hiddens = nn.ModuleList([
            nn.Linear(n_inputs // 2, n_inputs // 2)
            for _ in range(3-2)])
        self.output = nn.Linear(n_inputs // 2, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_domains=0):
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
            self.num_domains = num_domains

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

        # attention mask
        self.attention = CBAM(gate_channels=self.num_features, spatial=True)
        self.attention_1 = CBAM(gate_channels=self.num_features, spatial=True)

        # domain classifier
        self.domain_classifier = nn.Linear(self.num_features, self.num_domains)

        # domain discriminator
        self.domain_bn = nn.BatchNorm1d(self.num_features)
        self.domain_bn.bias.requires_grad_(False)
        self.domain_discriminator = nn.Linear(self.num_features, self.num_domains, bias=False)
        init.normal_(self.domain_discriminator.weight, std=0.001)
        # self.domain_discriminator = MLP(self.num_features, self.num_domains)

        # content discriminator
        self.content_bn = nn.BatchNorm1d(self.num_features)
        self.content_bn.bias.requires_grad_(False)
        self.content_discriminator = nn.Linear(self.num_features, self.num_classes, bias=False)
        init.normal_(self.content_discriminator.weight, std=0.001)
        # self.content_discriminator = MLP(self.num_features, self.num_classes)

        # batch norm bottleneck for domain classification
        self.dom_bottleneck = nn.BatchNorm1d(self.num_features)

        # bns

        if not pretrained:
            self.reset_params()

    @property
    def comm_modules(self):
        modules = []
        for m in self.children():
            if m not in self.adv_modules:
                modules.append(m)
        return modules

    @property
    def adv_modules(self):
        modules = [self.domain_discriminator, self.content_discriminator, self.content_bn, self.domain_bn]
        return modules

    def forward(self, x, output_prob=False):
        bs = x.size(0)
        # x = self.base(x)
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        c_mask, s_mask = self.attention(x)
        domain_feat = x * c_mask * s_mask
        # x = x * (1-c_mask) * (1-s_mask)
        c_mask_1, s_mask_1 = self.attention_1(x)
        content_feat = x * c_mask_1 * s_mask_1

        domain_disc_preds, content_disc_preds = self.forward_discriminator(domain_feat, content_feat)

        domain_preds = self.forward_domain(domain_feat=domain_feat)

        x = self.gap(content_feat)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False and output_prob is False):
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
            return prob, x, domain_preds, domain_disc_preds, content_disc_preds

    def forward_domain(self, domain_feat):
        domain_feat = self.gap(domain_feat)
        domain_feat = domain_feat[..., 0, 0]
        domain_feat = self.dom_bottleneck(domain_feat)
        domain_preds = self.domain_classifier(domain_feat)
        return domain_preds

    def forward_discriminator(self, domain_feat, content_feat):
        df = self.gap(domain_feat)[..., 0, 0]
        cf = self.gap(content_feat)[..., 0, 0]
        df = self.content_bn(df)
        cf = self.domain_bn(cf)

        domain_disc_preds = self.domain_discriminator(cf)
        content_disc_preds = self.content_discriminator(df)
        return domain_disc_preds, content_disc_preds

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


class ResNet_(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_domains=0, **kwargs):
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
            self.num_domains = num_domains

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

        # attention mask
        self.attention = CBAM(gate_channels=self.num_features, spatial=True)

        # domain classifier
        self.domain_classifier = nn.Linear(self.num_features, self.num_domains)

        # domain discriminator
        self.domain_discriminator = nn.Linear(self.num_features, self.num_domains)

        # content discriminator
        self.content_discriminator = nn.Linear(self.num_features, self.num_classes)

        # batch norm bottleneck for domain classification
        self.dom_bottleneck = nn.BatchNorm1d(self.num_features)

        # cross entropy and triplet loss
        self.ce_criterion = CrossEntropyLabelSmooth(self.num_classes)
        self.tri_criterion = TripletLoss(margin=0)

        if not pretrained:
            self.reset_params()

    def forward(self, inputs):
        x = inputs['images']
        id_labels = inputs['pids']
        domain_labels = inputs['domains']

        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # disentangle domain-relevant features and content-relevant features
        c_mask, s_mask = self.attention(x)
        domain_feat = x * c_mask * s_mask
        content_feat = x * (1-c_mask) * (1-s_mask)

        loss_dict = {}

        # domain classification branch
        loss_domain_cls = self.domain_classify(domain_feat, domain_labels)
        loss_dict.update(loss_domain_cls)

        # reid heads branch
        content_feat = self.gap(content_feat)
        content_feat = content_feat[..., 0, 0]
        if self.has_embedding:
            content_feat = self.feat_bn(self.feat(content_feat))
        else:
            content_feat = self.feat_bn(content_feat)


        if self.training:
            if self.has_embedding:
                content_feat = F.relu(content_feat)

            preds = self.classifier(content_feat)
            loss_dict['loss_cls'] = self.ce_criterion(preds, id_labels)
            loss_dict['loss_tri'] = self.tri_criterion(content_feat, id_labels)

            return preds, loss_dict

        else:
            content_feat = F.normalize(content_feat)
            return content_feat

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

    def domain_classify(self, domain_feat, domain_labels):
        domain_feat = self.gap(domain_feat)
        domain_feat = domain_feat[..., 0, 0]
        domain_feat = self.dom_bottleneck(domain_feat)
        domain_preds = self.domain_classifier(domain_feat)
        loss_dict = dict()
        loss_dict['loss_dom_cls'] = self.ce_criterion(domain_preds, domain_labels)
        return loss_dict



def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50_adv(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
