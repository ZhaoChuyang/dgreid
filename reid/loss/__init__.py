from __future__ import absolute_import

from .triplet import TripletLoss
from .triplet_xbm import TripletLossXBM
from .crossentropy import CrossEntropyLabelSmooth
from .idm_loss import DivLoss, BridgeFeatLoss, BridgeProbLoss
from .adv_loss import AdvLoss

__all__ = [
    'DivLoss',
    'BridgeFeatLoss',
    'BridgeProbLoss',
    'TripletLoss',
    'TripletLossXBM',
    'CrossEntropyLabelSmooth',
    'AdvLoss',
]
