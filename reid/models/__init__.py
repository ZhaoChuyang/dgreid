from __future__ import absolute_import

from .resnet import *
from .idm_module import *
from .resnet_idm import *
from .resnet_ibn import *
from .resnet_ibn_idm import *

from .resnet_adv import resnet50_adv
from .resnet_rsc import resnet50_rsc
from .resnet_attr import resnet50_attr, resnet50_attr_2
from .resnet_smm import resnet50_smm
from .resnet_smm_cy import resnet50_smm_cy
from .resnet_mde import resnet50_mde
from .resnet_mldg import resnet50_mldg
from .resnet_mldg_smm import resnet50_mldg_smm


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnet50_idm': resnet50_idm,
    'resnet_ibn50a_idm': resnet_ibn50a_idm,
    'resnet50_adv': resnet50_adv,
    'resnet50_rsc': resnet50_rsc,
    'resnet50_attr': resnet50_attr,
    'resnet50_attr_2': resnet50_attr_2,
    'resnet50_smm': resnet50_smm,
    'resnet50_smm_cy': resnet50_smm_cy,
    'resnet50_mde': resnet50_mde,
    'resnet50_mldg': resnet50_mldg,
    'resnet50_mldg_smm': resnet50_mldg_smm,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
