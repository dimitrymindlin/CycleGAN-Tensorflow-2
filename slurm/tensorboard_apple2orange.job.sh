#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir logs/apple2orange \
    --name "Apple2Orange Attention" \
    --description "Apple2Orange Attention" \
    --one_shot
