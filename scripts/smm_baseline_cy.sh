#!/usr/bin/env bash

# Baseline for Manifold SMM in two branch setting

# split by ','
source='dukemtmc,msmt17,cuhk03'
#source='market1501,dukemtmc'
target='market1501'
#target='dukemtmc'
#target='viper'
arch='resnet50_smm_cy'
epoch=90
batch_size=256
iter=200
step_size=30
smm_stage=1

# lam=0.5 => [0.5, 1]
# lam=0.3 => [0.7, 1]
lam=1

/usr/bin/python3 examples/smm_baseline_cy.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/DG_${arch}_baseline/${source}-TO-${target}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size} \
--smm-stage=${smm_stage} \
--lam=${lam} \
--combine-all \
#--rand-lam \
