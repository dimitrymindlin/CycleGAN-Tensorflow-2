#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir output/apple2orange/summaries/train \
    --name "AppleOrange Attention" \
    --description "AppleOrange Attention" \
    --one_shot
