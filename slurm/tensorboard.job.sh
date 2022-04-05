#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir output/apple2orange/summaries/train \
    --name "AppleOrange Attention" \
    --description "AppleOrange Attention" \
    --one_shot
