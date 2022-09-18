#!/usr/bin/env bash

python tune_mahalanobis_hyperparameter.py \
--name tune_mahalanobis \
--model_path checkpoints/pretrained_models/BiT-S-R101x1-flat-finetune.pth.tar \
--logdir checkpoints/finetune \
--datadir /imagenetpytorch/ \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--batch 32
