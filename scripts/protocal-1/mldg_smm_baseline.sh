#!/usr/bin/env bash
# split by ','
source='market1501,dukemtmc,cuhk02,cuhk03,cuhksysu'
target='prid,grid,viper,ilids'
arch='resnet50_mldg'
epoch=90
batch_size=56  # 128
iter=200
step_size=20  # 30
eval_step=10
mldg_beta=0.5
num_features=512

/usr/bin/python3 examples/mldg_smm_baseline_v2.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--eval-step ${eval_step} \
--mldg-beta ${mldg_beta} \
--combine-all \
#--features ${num_features} \
