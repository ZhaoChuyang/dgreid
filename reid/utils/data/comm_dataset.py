# encoding: utf-8
import numpy as np

from torch.utils.data import Dataset
from .base_dataset import BaseImageDataset

import scipy.io


class CommDataset(BaseImageDataset):
    """Image Person ReID Dataset, combine all datasets"""

    def __init__(self, img_items, verbose=True):
        super(CommDataset, self).__init__()
        self.img_items = img_items

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in img_items]
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        if verbose:
            print("=> Combine-all-reID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)


class DomainDataset(BaseImageDataset):
    """Image Person ReID Dataset, combine all datasets"""

    def __init__(self, img_items, verbose=True):
        super(DomainDataset, self).__init__()
        self.img_items = img_items

        pid_set = set()
        cam_set = set()
        domain_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            pid = i[1]
            domain = pid.split('_')[0]
            domain_set.add(domain)

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
        self.domain_dict = dict([(d, i) for i, d in enumerate(domain_set)])

        # train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in img_items]
        train = list()
        for img, pid, camid in img_items:
            domain = pid.split('_')[0]
            domain_id = self.domain_dict[domain]
            train.append((img, self.pid_dict[pid], self.cam_dict[camid], domain_id))

        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_domains = self.get_imagedata_info(self.train)
        if verbose:
            print("=> Combine-all-reID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def get_imagedata_info(self, data):
        pids, cams, doms = [], [], []
        for _, pid, camid, domid in data:
            pids += [pid]
            cams += [camid]
            doms += [domid]
        pids = set(pids)
        cams = set(cams)
        doms = set(doms)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_doms = len(doms)
        return num_pids, num_imgs, num_cams, num_doms

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_domains = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_domains = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_domains = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class AttrDataset(BaseImageDataset):
    """Image Person ReID Dataset, combine all datasets"""

    def __init__(self, img_items, verbose=True):
        super(AttrDataset, self).__init__()
        self.img_items = img_items

        pid_set = set()
        cam_set = set()
        self.attribute_dict = self.get_attribute_dict()

        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        train = [(img, self.pid_dict[pid], self.cam_dict[camid], self.attribute_dict[pid]) for img, pid, camid in img_items]
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        if verbose:
            print("=> Combine-all-reID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def get_imagedata_info(self, data):
        pids, cams, doms = [], [], []
        for _, pid, camid, attributes in data:
            pids += [pid]
            cams += [camid]
            # doms += [domid]
        pids = set(pids)
        cams = set(cams)
        doms = set(doms)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        # num_doms = len(doms)
        return num_pids, num_imgs, num_cams

    def get_attribute_dict(self):
        attribute_dict = dict()
        num_attributes = 18

        mdict = scipy.io.loadmat('/data/dgreid/attributes/market/market_attribute.mat')
        train_attrs = mdict['market_attribute'][0, 0]['train']
        test_attrs = mdict['market_attribute'][0, 0]['test']
        attribute_dict.update(self.get_market_attributes(train_attrs, 'train'))
        attribute_dict.update(self.get_market_attributes(test_attrs, 'test'))
        # attribute_dict.update({'market1501_test_0': np.zeros(num_attributes)})

        mdict = scipy.io.loadmat('/data/dgreid/attributes/duke/duke_attribute.mat')
        train_attrs = mdict['duke_attribute'][0, 0]['train']
        test_attrs = mdict['duke_attribute'][0, 0]['test']
        attribute_dict.update(self.get_duke_attributes(train_attrs, 'train'))
        attribute_dict.update(self.get_duke_attributes(test_attrs, 'test'))
        # attribute_dict.update({'market1501_test_0': np.zeros(num_attributes)})

        return attribute_dict

    def get_market_attributes(self, arr, mode):
        indices = arr['image_index'][0, 0][0]
        # attribute_names = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray',
        #                    'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue',
        #                    'upgreen',
        #                    'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down', 'up', 'hair', 'hat',
        #                    'gender']
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
            attr_val = arr[attr_name][0, 0][0] - 1
            attributes[:, idx] = attr_val

        if mode == 'train':
            for idx, pid in enumerate(indices):
                attribute_dict['market1501_{}'.format(int(pid))] = attributes[idx]
        elif mode == 'test':
            for idx, pid in enumerate(indices):
                attribute_dict['market1501_test_{}'.format(int(pid))] = attributes[idx]

        return attribute_dict

    def get_duke_attributes(self, arr, mode):
        indices = arr['image_index'][0, 0][0]
        # attribute_names = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray',
        #                    'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue',
        #                    'upgreen',
        #                    'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down', 'up', 'hair', 'hat',
        #                    'gender']
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
            attr_val = arr[attr_name][0, 0][0] - 1
            attributes[:, idx] = attr_val

        if mode == 'train':
            for idx, pid in enumerate(indices):
                attribute_dict['dukemtmc_{}'.format(int(pid))] = attributes[idx]
        elif mode == 'test':
            for idx, pid in enumerate(indices):
                attribute_dict['dukemtmc_test_{}'.format(int(pid))] = attributes[idx]

        return attribute_dict


