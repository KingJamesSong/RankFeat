#!/bin/bash

METHOD=RankFeat       #OOD Method
OUT_DATA=iNaturalist  #OOD Dataset

python test_ood_squeezenet.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /data/datasets/imagenet/val \
--out_datadir /data/datasets/${OUT_DATA} \
--batch 32 \
--logdir checkpoints/test_log \
--score ${METHOD} \
