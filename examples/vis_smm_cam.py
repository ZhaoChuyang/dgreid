from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta
import tabulate

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2

sys.path.append(".")
from reid import datasets
from reid import models
# from reid.models.dsbn import convert_dsbn, convert_bn
# from reid.models.csbn import convert_csbn
# from reid.models.idm_dsbn import convert_dsbn_idm, convert_bn_idm
# from reid.models.xbm import XBM
from reid.trainers import MLDGSMMTrainer3
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import CommDataset
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.solver import WarmupMultiStepLR
from reid.utils.tools import mkdir_if_missing
from reid.utils.serialization import load_checkpoint, copy_state_dict

from PIL import Image
from reid.models.layers.adain import adaptive_instance_normalization_v2
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
    model,
    test_loader,
    save_dir,
    width,
    height,
    use_gpu,
    target,
    img_mean=None,
    img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    data_loader = test_loader
    # original images and activation maps are saved individually
    actmap_dir = osp.join(save_dir, 'actmap_' + target)
    mkdir_if_missing(actmap_dir)
    print('Visualizing activation maps for {} ...'.format(target))

    for batch_idx, data in enumerate(data_loader):
        imgs, paths = data['images'], data['fnames']
        if use_gpu:
            imgs = imgs.cuda()

        # forward to get convolutional feature maps
        try:
            [ori_outputs, smm_outputs] = model(imgs, return_featuremaps=True)
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )

        if ori_outputs.dim() != 4 or smm_outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {}, {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    ori_outputs.dim(), smm_outputs.dim()
                )
            )

        # compute activation maps
        ori_outputs = (ori_outputs**2).sum(1)
        b, h, w = ori_outputs.size()
        ori_outputs = ori_outputs.view(b, h * w)
        ori_outputs = F.normalize(ori_outputs, p=2, dim=1)
        ori_outputs = ori_outputs.view(b, h, w)

        smm_outputs = (smm_outputs**2).sum(1)
        b, h, w = smm_outputs.size()
        smm_outputs = smm_outputs.view(b, h * w)
        smm_outputs = F.normalize(smm_outputs, p=2, dim=1)
        smm_outputs = smm_outputs.view(b, h, w)

        if use_gpu:
            imgs, ori_outputs, smm_outputs = imgs.cpu(), ori_outputs.cpu(), smm_outputs.cpu()

        for j in range(ori_outputs.size(0)):
            # get image name
            path = paths[j]
            imname = osp.basename(osp.splitext(path)[0])

            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

            # activation original map
            ori_am = ori_outputs[j, ...].numpy()
            ori_am = cv2.resize(ori_am, (width, height))
            ori_am = 255 * (ori_am - np.min(ori_am)) / (
                np.max(ori_am) - np.min(ori_am) + 1e-12
            )
            ori_am = np.uint8(np.floor(ori_am))
            ori_am = cv2.applyColorMap(ori_am, cv2.COLORMAP_JET)

            # overlapped
            ori_overlapped = img_np*0.3 + ori_am*0.7
            ori_overlapped[ori_overlapped > 255] = 255
            ori_overlapped = ori_overlapped.astype(np.uint8)

            # smm activation map
            smm_am = smm_outputs[j, ...].numpy()
            smm_am = cv2.resize(smm_am, (width, height))
            smm_am = 255 * (smm_am - np.min(smm_am)) / (
                np.max(smm_am) - np.min(smm_am) + 1e-12
            )
            smm_am = np.uint8(np.floor(smm_am))
            smm_am = cv2.applyColorMap(smm_am, cv2.COLORMAP_JET)

            # overlapped
            smm_overlapped = img_np*0.3 + smm_am*0.7
            smm_overlapped[smm_overlapped > 255] = 255
            smm_overlapped = smm_overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
                     width + GRID_SPACING:2*width + GRID_SPACING, :] = ori_overlapped
            grid_img[:, 2*width + 2*GRID_SPACING:, :] = smm_overlapped
            cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

        if (batch_idx+1) % 10 == 0:
            print(
                '- done batch {}/{}'.format(
                    batch_idx + 1, len(data_loader)
                )
            )

        # if batch_idx > 100:
        #     break


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=False, dropout=args.dropout,
                          num_classes=args.nclass)

    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def get_data(name, data_dir, combineall=False):
    # data_dir = '/data/datasets'
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, combineall=combineall)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers, num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    preprocessor = Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer)
    loader = DataLoader(preprocessor,
                        batch_size=batch_size,
                        num_workers=workers,
                        sampler=sampler,
                        shuffle=not rmgs_flag,
                        pin_memory=False,
                        drop_last=True)
    train_loader = IterLoader(loader, length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader


if __name__ == '__main__':
    arch = 'resnet50_mldg_smm'
    # arch = 'resnet50'
    num_features = 0
    dropout = 0
    num_classes = 0
    # weight_path = '/data/IDM_logs/DG_resnet50_baseline/dukemtmc,msmt17,cuhk03-TO-market1501-epo90-step30-iter200-batch256/model_best.pth.tar'
    weight_path = '/data/IDM_logs/DG_resnet50_mldg_baseline/dukemtmc,msmt17,cuhk03-TO-market1501-epo90-step20-iter200-batch120/model_best.pth.tar'
    use_gpu = True
    width = 64
    height = 128
    target_dataset_name = 'market1501'
    save_dir = 'vis/act_smm_cam/%s_%s' % (target_dataset_name, arch)

    model = models.create(arch, num_features=num_features, norm=False, dropout=dropout, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.cuda()

    checkpoint = load_checkpoint(weight_path)['state_dict']
    model = copy_state_dict(checkpoint, model)

    target_dataset = get_data(target_dataset_name, '/data/datasets')
    target_loader = get_test_loader(target_dataset, height, width, batch_size=32, workers=0)

    visactmap(model, target_loader, save_dir, width, height, use_gpu, target_dataset_name)



