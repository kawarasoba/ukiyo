#!/bin/bash

python train.py \
    -c='test' \
    -m='resnet50' \
    -e=5 \
    --batch_size=32 \
    -p='params' \
    -l='logs' \
    -ti='data' \
    -tl='data'

#python train.py \
#    -c='seres_oversampling' \
#    -m='se_resnet50' \
#    -e=1000 \
#    -result_path='result' \
#    -result_case='dense+effi+se+seres+ince_ratio3' \
#    --confidence_border=0.892 \
#    --over_sampling=True \
#    --mixup=True \
#    -p='params' \
#    -l='logs' \
#    -ti='data' \
#    -tl='data' \
#    -test_i='data'

#<< local
#python train.py \
#    -c='case_name' \
#    -m='model_name' \
#    -e=1000 \
#    --over_sampling=False \
#    --add_class_weight=False \
#    --mixup=True \
#    --augmix=False \
#    --aug_decrease=False \
#    --nfold=0 \
#    --restart_epoch=0 \
#    --restart_from_final=False
#local

#<< kqi
#python train.py \
#    -c='case_name' \
#    -m='model_name' \
#    -e=1000 \
#    --over_sampling=False \
#    --add_class_weight=False \
#    --mixup=True \
#    --augmix=False \
#    --aug_decrease=False \
#    -p='/kqi/output/params' \
#    -l='/kqi/output/logs' \
#    -ti='/kqi/input/training/22465713' \
#    -tl='/kqi/input/training/22465714' \
#    --nfold=0 \
#    --restart_epoch=0 \
#    --restart_from_final=False
#kqi