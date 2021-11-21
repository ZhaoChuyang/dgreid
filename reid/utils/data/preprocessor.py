from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch
import pdb
import scipy.io
from ...models.layers.adain import adaptive_instance_normalization_v2


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {
            'images': img,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
        }



class StylePreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(StylePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mean = torch.tensor([-0.4414, -0.4257, -0.2294]).unsqueeze(1).unsqueeze(2).repeat(1, 256, 128)
        self.std = torch.tensor([0.9254, 0.9121, 0.8947]).unsqueeze(1).unsqueeze(2).repeat(1, 256, 128)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # from IPython import embed
        # embed()
        # print(self.mean.shape, self.std.shape)
        mean, std = self.calc_mean_std(img.unsqueeze(0))
        mean = mean[0]
        std = std[0]
        size = img.shape
        img = ((img - mean.expand(size)) / (std.expand(size))) * self.std + self.mean

        return {
            'images': img,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
        }


class DomainPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(DomainPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, domain_id = self.dataset[index]

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {
            'images': img,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
            'domains': domain_id
        }


class MoCoProcessor(Dataset):
    def __init__(self, dataset, root=None, transform_q=None, transform_k=None):
        super(MoCoProcessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform_q = transform_q
        self.transform_k = transform_k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        try:
            fname, pid, camid, domain_id = self.dataset[index]
        except:
            fname, pid, camid = self.dataset[index]
            domain_id = 0

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        im_q = im_k = img
        if self.transform_q is not None:
            im_q = self.transform_q(img)
        if self.transform_k is not None:
            im_k = self.transform_k(img)

        return {
            'im_q': im_q,
            'im_k': im_k,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
            'domains': domain_id
        }


class AttrPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(AttrPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, attributes = self.dataset[index]

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {
            'images': img,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
            'attributes': torch.tensor(attributes).long()
        }


class MixedStylePreprocessor(Dataset):
    def __init__(self, dataset_cont, dataset_styl, root=None, transform=None):
        super(MixedStylePreprocessor, self).__init__()
        self.dataset_cont = dataset_cont
        self.dataset_styl = dataset_styl
        self.root = root
        self.transform = transform
        self.len_style_set = len(self.dataset_styl)

    def __len__(self):
        return len(self.dataset_cont)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset_cont[index]
        s_fname, _, _ = self.dataset_styl[index % self.len_style_set]

        # fname, pid, camid = self.dataset[index]
        fpath = fname
        s_fpath = s_fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
            s_fpath = osp.join(self.root, s_fpath)

        img = Image.open(fpath).convert('RGB')
        s_img = Image.open(s_fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            s_img = self.transform(s_img)

        img = adaptive_instance_normalization_v2(torch.unsqueeze(img, 0), torch.unsqueeze(s_img, 0), 1)
        img = torch.squeeze(img)

        return {
            'images': img,
            'fnames': fname,
            'pids': pid,
            'camids': camid,
            'indices': index,
        }


def get_market_attributes(arr):
    indices = arr['image_index'][0,0][0]

    attribute_names = ['backpack',
                       'bag',
                       'handbag',
                       'gender',
                       'hat',
                       'downblack',
                       'downwhite',
                       'downgray',
                       'downblue',
                       'downgreen',
                       'downbrown',
                       'upblack',
                       'upwhite',
                       'upred',
                       'uppurple',
                       'upgray',
                       'upblue',
                       'upgreen']
    num_pids = len(indices)
    num_attrs = len(attribute_names)
    attribute_dict = {}
    attributes = np.zeros(shape=(num_pids, num_attrs))
    for idx, attr_name in enumerate(attribute_names):
        # NOTE: make 0 the minimal value
        attr_val = arr[attr_name][0,0][0] - 1
        attributes[:, idx] = attr_val

    for idx, pid in enumerate(indices):
        attribute_dict[int(pid)] = attributes[idx]

    return attribute_dict

if __name__ == '__main__':
    mdict = scipy.io.loadmat('/data/dgreid/attributes/duke/duke_attribute.mat')
    train_attrs = mdict['duke_attribute'][0, 0]['train']
    test_attrs = mdict['duke_attribute'][0, 0]['test']

    attribute_dict = {}
    attribute_dict.update(get_market_attributes(train_attrs))
    attribute_dict.update(get_market_attributes(test_attrs))

    print(attribute_dict)



    # mdict = scipy.io.loadmat('/data/dgreid/attributes/market/market_attribute.mat')
    # train_attrs = mdict['market_attribute'][0, 0]['train']
    # test_attrs = mdict['market_attribute'][0, 0]['test']
    #
    # # attribute_names = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down', 'up', 'hair', 'hat', 'gender']
    # attribute_dict = {}
    # attribute_dict.update(get_market_attributes(train_attrs))
    # attribute_dict.update(get_market_attributes(test_attrs))
    #
    # num_attributes = 18
    #
    # attribute_dims = [None] * num_attributes
    #
    # temp = 0
    #
    # for idx in range(num_attributes):
    #     minval = 1000
    #     maxval = 0
    #     for key, val in attribute_dict.items():
    #         minval = min(minval, val[idx])
    #         maxval = max(maxval, val[idx])
    #     attribute_dims[idx] = int(maxval - minval + 1)
    #
    # # print(attribute_dims)
    # print(attribute_dims)
