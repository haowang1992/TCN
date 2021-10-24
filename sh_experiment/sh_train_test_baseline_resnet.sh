#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=`pwd` /home/wanghao/anaconda3/envs/pt1.4/bin/python experiment/train_baseline_resnet.py \
--dataset_name Sketchy \
--zero_version zeroshot1 \
--backbone_ncls 100 \
--backbone_nhash 64 \
--sbh \
--lambda_sbh 1.0 --arch baseline_resnet

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=`pwd` /home/wanghao/anaconda3/envs/pt1.4/bin/python experiment/train_baseline_resnet.py \
--dataset_name TUBerlin \
--zero_version zeroshot \
--backbone_ncls 220 \
--backbone_nhash 64 \
--sbh \
--lambda_sbh 1.0 --arch baseline_resnet


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=`pwd` /home/wanghao/anaconda3/envs/pt1.4/bin/python experiment/train_baseline_resnet.py \
--dataset_name Sketchy \
--zero_version zeroshot1 \
--backbone_ncls 100 \
--backbone_nhash 64 \
--sbh \
--lambda_sbh 1.0 --arch baseline_resnet --testing

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=`pwd` /home/wanghao/anaconda3/envs/pt1.4/bin/python experiment/train_baseline_resnet.py \
--dataset_name TUBerlin \
--zero_version zeroshot \
--backbone_ncls 220 \
--backbone_nhash 64 \
--sbh \
--lambda_sbh 1.0 --arch baseline_resnet --testing



