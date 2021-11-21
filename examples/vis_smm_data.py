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
import os

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

from PIL import Image
from reid.models.layers.adain import adaptive_instance_normalization_v2
import matplotlib.pyplot as plt


def relabel_datasets(data):
    num_pids = 0
    num_camids = 0
    for dataset in data:
        train = [(img, pid + num_pids, camid + num_camids) for img, pid, camid in dataset.train]
        dataset.train = train
        num_pids += dataset.num_train_pids
        num_camids += dataset.num_train_cams
    print('Totally %d pids, %d camids' % (num_pids, num_camids))


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

    train_loaders = []
    for data in dataset:
        train_set = sorted(data.train) if trainset is None else sorted(trainset)
        rmgs_flag = num_instances > 0
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None

        preprocessor = Preprocessor(train_set, root=data.images_dir, transform=train_transformer)
        dataloader = DataLoader(preprocessor,
                                batch_size=batch_size,
                                num_workers=workers,
                                sampler=sampler,
                                shuffle=not rmgs_flag,
                                pin_memory=False,
                                drop_last=True)
        train_loader = IterLoader(dataloader, length=iters)
        train_loaders.append(train_loader)

    return train_loaders


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader


def main():
    content_dataset_name = 'market1501'
    style_dataset_name = 'dukemtmc,dukemtmc,dukemtmc,dukemtmc'

    style_datasets = []
    content_dataset = get_data(content_dataset_name, '/data/datasets', True)
    for dataset_name in style_dataset_name.split(','):
        style_dataset = get_data(dataset_name, '/data/datasets', True)
        style_dataset = CommDataset(style_dataset.train)
        random.shuffle(style_dataset.train)
        style_datasets.append(style_dataset)

    content_dataset = CommDataset(content_dataset.train)
    random.shuffle(content_dataset.train)

    num_cols = len(style_datasets)
    num_rows = 5

    transformer = T.Compose([
             T.Resize((256, 128), interpolation=3),
             T.ToTensor(),
         ])

    fig, axs = plt.subplots(2, 2)

    style_images_path = 'duke-images'
    files = os.listdir(style_images_path)

    for row_id in range(num_rows):
        for col_id in range(len(files)):
            content_img_path = content_dataset.train[row_id][0]
            # style_img_path = style_datasets[col_id].train[col_id][0]
            style_img_path = '%s/%s' % (style_images_path, files[col_id])
            content_img = Image.open(content_img_path)
            style_img = Image.open(style_img_path)
            content_img_ten = transformer(content_img).unsqueeze(0)
            style_img_ten = transformer(style_img).unsqueeze(0)
            stylized_img_ten = adaptive_instance_normalization_v2(content_img_ten, style_img_ten, 1)
            stylized_img_ten = stylized_img_ten[0, ...]
            # print(stylized_img_ten.shape)
            stylized_img_ten = stylized_img_ten.permute([1, 2, 0])
            stylized_img_ten = stylized_img_ten - torch.min(stylized_img_ten)
            stylized_img_ten = stylized_img_ten / torch.max(stylized_img_ten)
            # print(torch.max(stylized_img_ten), torch.min(stylized_img_ten))

            # plt.savefig()

            content_img = content_img.resize((128, 256))
            style_img = style_img.resize((128, 256))

            axs[0, 1].imshow(style_img)
            axs[1, 0].imshow(content_img)
            axs[1, 1].imshow(stylized_img_ten)
            # fig.savefig('vis/smm_images/test.png')
            plt.imsave('vis/smm_images/%d_%d.png' % (row_id, col_id), stylized_img_ten)
            plt.imsave('vis/smm_images/row_%d.png' % (row_id), content_img)
            plt.imsave('vis/smm_images/col_%d.png' % (col_id), style_img)





if __name__ == '__main__':
    main()
