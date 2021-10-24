#!/usr/bin/env bash
source='dukemtmc'
target='market1501'
arch='resnet50'
epoch='80'
csdn='True'
python3 examples/train_baseline.py -ds ${source} -dt ${target} -a ${arch} --epochs ${epoch} \
--csdn ${csdn} \
--logs-dir /data/IDM_logs/${arch}_csdn_baseline/${source}-TO-${target}-epo${epoch}
