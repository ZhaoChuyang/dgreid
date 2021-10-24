#!/usr/bin/env bash
# split by ','
#source='dukemtmc,msmt17,cuhk03'
source='market1501,dukemtmc'
#target='market1501'
target='viper'
#arch='resnet50_attr'
arch='resnet50_attr_2'
epoch=120
batch_size=256
#iter=200
iter=100
step_size=30
lam=0.9


/usr/bin/python3 examples/attr_baseline.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--lam ${lam} \
#--combine-all \
