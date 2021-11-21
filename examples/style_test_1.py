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

sys.path.append(".")
from reid import datasets
from reid import models
# from reid.models.dsbn import convert_dsbn, convert_bn
# from reid.models.csbn import convert_csbn
# from reid.models.idm_dsbn import convert_dsbn_idm, convert_bn_idm
# from reid.models.xbm import XBM
from reid.trainers import Baseline_Trainer, IDM_Trainer, Base_Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import CommDataset
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor, StylePreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.rerank import compute_jaccard_distance


start_epoch = best_mAP = 0

mean = torch.tensor([-0.44, -0.43, -0.23])
std = torch.tensor([0.99, 0.96, 0.93])


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


def get_test_loader(dataset, height, width, batch_size, workers, testset=None, stylized=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    if stylized:
        processor = StylePreprocessor(testset, root=dataset.images_dir, transform=test_transformer)
    else:
        processor = Preprocessor(testset, root=dataset.images_dir, transform=test_transformer)

    test_loader = DataLoader(
        processor,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_mean_std(dataloader):
    num_batches = len(dataloader)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for iter in range(num_batches):
        mini_batch = dataloader.next()['images']
        b_mean, b_std = calc_mean_std(mini_batch)

        mean += torch.mean(b_mean, dim=[0, 2, 3])
        std += torch.mean(b_std, dim=[0, 2, 3])
    mean /= num_batches
    std /= num_batches


    # mean /= num_batches
    #
    # dataloader.new_epoch()
    # mean = mean.unsqueeze(1)
    # mean = mean.unsqueeze(2)
    # mean = mean.repeat(1, 256, 128)
    # for iter in range(num_batches):
    #     mini_batch = dataloader.next()['images']
    #     std += torch.mean((mini_batch - mean) ** 2, dim=[0, 2, 3])
    # std /= num_batches

    return mean, std


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=False, dropout=args.dropout, 
                          num_classes=args.nclass)

    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain dataset")
    train_items = []
    for src in args.dataset_source.split(','):
        dataset = get_data(src, args.data_dir, args.combine_all)
        train_items.extend(dataset.train)
    dataset_source = CommDataset(train_items)

    print("==> Load target-domain dataset")
    target_loaders = []
    target_datasets = []
    target_dataset_names = args.dataset_target.split(',')

    for target_dataset_name in target_dataset_names:
        target_dataset_names = target_dataset_name.split('_')
        if len(target_dataset_names) != 1:
            target_dataset_name = target_dataset_names[0]
            target_dataset = get_data(target_dataset_name, args.data_dir)
            target_loader = get_test_loader(target_dataset, args.height, args.width, args.batch_size, args.workers, stylized=True)
            target_loaders.append(target_loader)
            target_datasets.append(target_dataset)
        else:
            target_dataset = get_data(target_dataset_name, args.data_dir)
            target_loader = get_test_loader(target_dataset, args.height, args.width, args.batch_size, args.workers)
            target_loaders.append(target_loader)
            target_datasets.append(target_dataset)

    # dataset_target = get_data(args.dataset_target, args.data_dir)
    # test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                           args.batch_size, args.workers, args.num_instances, iters)

    source_classes = dataset_source.num_train_pids

    # mean, std = get_mean_std(train_loader_source)

    # from IPython import embed
    # embed()

    args.nclass = source_classes

    # Create model
    model = create_model(args)
    print(model)

    # Evaluator
    evaluator = Evaluator(model)
    best_mAP = [0] * len(target_datasets)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 20, 40, 60], gamma=0.1)

    # Trainer
    trainer = Base_Trainer(model, args.nclass, margin=args.margin)

    table = []
    header = ['Epoch', 'Dataset', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10']
    table.append(header)

    for epoch in range(args.epochs):

        # train_loader_source.new_epoch()
        # train_loader_target.new_epoch()
        trainer.train(epoch, train_loader_source, optimizer, print_freq=args.print_freq, train_iters=args.iters)
                      
        if (epoch+1) % args.eval_step == 0 or (epoch == args.epochs-1):
            for target_id in range(len(target_datasets)):
                target_dataset_name = target_dataset_names[target_id]
                target_dataset = target_datasets[target_id]
                target_loader = target_loaders[target_id]
                print('Test on target: ', target_dataset_name)
                result_dict, mAP = evaluator.evaluate(target_loader, target_dataset.query, target_dataset.gallery,
                                                      cmc_flag=True)

                # show results in table
                record = list()
                record.append(epoch)
                record.append(target_dataset_name)
                record.append(result_dict['mAP'])
                record.append(result_dict['rank-1'])
                record.append(result_dict['rank-5'])
                record.append(result_dict['rank-10'])
                table.append(record)

                print(tabulate.tabulate(table, headers='firstrow', tablefmt='github', floatfmt='.2%'))

                is_best = mAP > best_mAP[target_id]
                best_mAP[target_id] = max(mAP, best_mAP[target_id])

                # save_checkpoint({
                #     'state_dict': model.state_dict(),
                #     'epoch': epoch + 1,
                #     'best_mAP': best_mAP,
                # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP[target_id], ' *' if is_best else ''))

            # print('Test on target: ', args.dataset_target)
            # result_dict, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            #
            # # show results in table
            # record = []
            # record.append(epoch)
            # record.append(target_dataset_name)
            # record.append(result_dict['mAP'])
            # record.append(result_dict['rank-1'])
            # record.append(result_dict['rank-5'])
            # record.append(result_dict['rank-10'])
            # table.append(record)
            #
            # print(tabulate.tabulate(table, headers='firstrow', tablefmt='github', floatfmt='.2%'))
            #
            # is_best = (mAP > best_mAP)
            # best_mAP = max(mAP, best_mAP)
            # save_checkpoint({
            #     'state_dict': model.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            #
            # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
            #       format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()
    # print ('==> Test with the best model on the target domain:')
    # checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    # model.load_state_dict(checkpoint['state_dict'])
    # evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc')
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
    parser.add_argument('--combine-all', action='store_true',
                        help="if True: combinall train, query, gallery for training;")
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--nclass', type=int, default=1000,
                        help="number of classes (source+target)")
    parser.add_argument('--s-class', type=int, default=1000,
                        help="number of classes (source)")
    parser.add_argument('--t-class', type=int, default=1000,
                        help="number of classes (target)")
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--mu1', type=float, default=0.5,
                        help="weight for loss_bridge_pred")
    parser.add_argument('--mu2', type=float, default=0.1,
                        help="weight for loss_bridge_feat")
    parser.add_argument('--mu3', type=float, default=1,
                        help="weight for loss_div")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_idm',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    # xbm parameters
    parser.add_argument('--memorySize', type=int, default=8192,
                        help='meomory bank size')
    parser.add_argument('--ratio', type=float, default=1,
                        help='memorySize=ratio*data_size')
    parser.add_argument('--featureSize', type=int, default=2048)
    parser.add_argument('--use-xbm', action='store_true', help="if True: strong baseline; if False: naive baseline")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=30)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=10)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, default='/data/datasets')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    # hbchen
    parser.add_argument('--csdn', type=bool, default=False)
    main()

