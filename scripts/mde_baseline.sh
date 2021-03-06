#!/usr/bin/env bash
# split by ','
source='dukemtmc,msmt17,cuhk03'
#source='market1501,dukemtmc'
target='market1501'
#target='dukemtmc'
#target='viper'
arch='resnet50_mde'
epoch=90
batch_size=64  # 128
iter=200
step_size=20  # 30

/usr/bin/python3 examples/mde_baseline.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--combine-all \
