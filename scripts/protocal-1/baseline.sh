#!/usr/bin/env bash
# split by ','
source='market1501,dukemtmc,cuhk02,cuhk03,cuhksysu'
target='prid,grid,viper,ilids'
arch='resnet50'
epoch=90
batch_size=64
iter=200
step_size=40
num_instances=4

#epoch=90
#batch_size=256
#iter=200
#step_size=30
#num_instances=4

/usr/bin/python3 examples/DG_baseline.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--num-instances ${num_instances} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--combine-all \
