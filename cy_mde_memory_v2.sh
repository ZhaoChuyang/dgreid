#!/usr/bin/env bash
# split by ','
source='dukemtmc,msmt17,cuhk03'
#source='dukemtmc'
#source='market1501'
target='market1501'
arch='resnet50_mde'
epoch='90'
iter='200'
step_size='20'
batch_size='64'
ids='8'
dim='0'
mm='0.35'
bn='BN'
/usr/bin/python3 examples/MDE_memory_v2.py -ds ${source} \
-dt ${target} \
-a ${arch} \
--num-instances ${ids} \
--features ${dim} \
--mem-margin ${mm} \
--momentum '0.1' \
--bn-type ${bn} \
--epochs ${epoch} -b ${batch_size} --iters ${iter} --step-size ${step_size} \
--logs-dir /data/IDM_logs/mde_memory_v2/test2 \
--bn-type ${bn} \
#-cls \
#--mean-update \
#-ibn \
#--combine-all \


# base-BN-${bn}-mem-margin${mm}-ids${ids}-dim${dim}-epo${epoch}-step${step_size}-iter${iter}-batch${batch_size}
