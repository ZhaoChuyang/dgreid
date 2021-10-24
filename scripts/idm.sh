#!/usr/bin/env bash
source='dukemtmc'
target='market1501'
arch='resnet50_idm'
stage=0
mu1='0.7'
mu2='0.1'
mu3='1.0'
python3 examples/train_idm.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir /data/IDM_logs/logs/${arch}/${source}-TO-${target} \
--stage ${stage} --mu1 ${mu1} --mu2 ${mu2} --mu3 ${mu3}
