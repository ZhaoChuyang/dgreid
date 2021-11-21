#!/usr/bin/env bash
# split by ','
source='market1501'
#source='market1501,dukemtmc'
target='dukemtmc_style'
#target='dukemtmc'
#target='viper'
arch='resnet50'
epoch=90
batch_size=64
iter=200
step_size=30

/usr/bin/python3 examples/style_test_1.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--combine-all \
