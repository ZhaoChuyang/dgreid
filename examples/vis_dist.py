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

from tqdm import tqdm
from PIL import Image
from reid.models.layers.adain import adaptive_instance_normalization_v2
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import seaborn as sns


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10





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


def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, trainset=None):

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
        dataset.query = [tuple(r) for r in dataset.query]
        dataset.gallery = [tuple(r) for r in dataset.gallery]
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader


@torch.no_grad()
def vis_multi_dist(model, loaders, names):
    model.eval()
    all_features = []

    tsne = TSNE(n_components=2)

    for loader in loaders:
        features = None
        for batch in tqdm(loader, total=len(loader)):
            images = batch['images']
            feat = model(images)
            if features is None:
                features = feat
            features = torch.cat([features, feat], dim=0)

            if len(features) > 1000:
                break

        all_features.append(features)

    dist_id = []
    for idx, features in enumerate(all_features):
        dist_id += [idx] * len(features)

    all_features = torch.cat(all_features, dim=0)
    all_features = all_features.cpu().numpy()

    print('run t-sne...')
    tsne_features = tsne.fit_transform(all_features)

    # sns_plot = sns.scatterplot(tsne_features[:, 0], tsne_features[:, 1], hue=dist_id, markers=[',', ','])
    dist_id = np.array(dist_id)
    all_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(len(loaders)):
        # if i == 0:
        #     color = 'b'
        # else:
        #     color = 'r'
        indices = np.argwhere(dist_id == i).flatten()
        plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], c=all_colors[i], marker='o', s=2, linewidths=0, label=names[i])
    # plt.legend()
    # fig = sns_plot.get_figure()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('vis/dist_fig_resnet_resnet.png')
    #
    # from IPython import embed
    # embed()

if __name__ == '__main__':
    # arch = 'resnet50_mldg'
    arch = 'resnet50'
    num_features = 0
    dropout = 0
    num_classes = 0
    weight_path = '/data/IDM_logs/DG_resnet50_baseline/dukemtmc,msmt17,cuhk03-TO-market1501-epo90-step30-iter200-batch256/model_best.pth.tar'
    # weight_path = '/data/IDM_logs/DG_resnet50_mldg_baseline/dukemtmc,msmt17,cuhk03-TO-market1501-epo90-step20-iter200-batch120/model_best.pth.tar'
    use_gpu = True
    width = 64
    height = 128
    target_dataset_name = 'market1501'
    train_dataset_name = 'dukemtmc'

    save_dir = 'vis/act_cam/%s_%s' % (target_dataset_name, arch)

    model = models.create(arch, num_features=num_features, norm=False, dropout=dropout, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.cuda()

    checkpoint = load_checkpoint(weight_path)['state_dict']
    model = copy_state_dict(checkpoint, model)

    target_dataset = get_data(target_dataset_name, '/data/datasets')
    target_loader = get_test_loader(target_dataset, height, width, batch_size=32, workers=0)

    train_dataset = get_data(train_dataset_name, '/data/datasets')
    train_loader = get_test_loader(train_dataset, height, width, batch_size=32, workers=0)

    train_dataset_1 = get_data('cuhk03', '/data/datasets')
    train_loader_1 = get_test_loader(train_dataset_1, height, width, batch_size=32, workers=0)

    train_dataset_2 = get_data('msmt17', '/data/datasets')
    train_loader_2 = get_test_loader(train_dataset_2, height, width, batch_size=32, workers=0)

    vis_multi_dist(model, [target_loader, train_loader, train_loader_2, train_loader_1], ['market', 'duke', 'msmt', 'cuhk03'])
