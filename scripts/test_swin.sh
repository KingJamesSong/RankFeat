#!/bin/bash

METHOD=$1       #OOD Method
OUT_DATA=$2     #OOD Dataset
BATCH_SIZE=$3   #OOD test batch size
CUDA_ID=$4      #CUDA index

python test_ood_swin.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /data/imagenet/images/val \
--out_datadir /data/${OUT_DATA} \
--batch ${BATCH_SIZE} \
--logdir checkpoints/test_log \
--score ${METHOD} \
--workers 8 \
--cuda_id ${CUDA_ID} 