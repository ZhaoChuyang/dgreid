#!/usr/bin/env bash
# split by ','
source='dukemtmc,msmt17,cuhk03'
#source='market1501'
target='market1501'
arch='resnet50_adv'
epoch='100'
batch_size='256'
iter='300'
step_size='40'
adv_update_steps=10
lr=0.00035

/usr/bin/python3 examples/adv_baseline.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--lr ${lr} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--adv_update_steps ${adv_update_steps} \
--combine-all \
