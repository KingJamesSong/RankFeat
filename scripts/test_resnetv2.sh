#!/bin/bash

METHOD=RankFeat       #OOD Method
OUT_DATA=iNaturalist  #OOD Dataset

python test_ood_resnetv2.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /data/datasets/imagenet/val \
--out_datadir /data/datasets/${OUT_DATA} \
--model_path BiT-S-R101x1-flat-finetune.pth.tar \
--batch 32 \
--logdir checkpoints/test_log \
--score ${METHOD} \
