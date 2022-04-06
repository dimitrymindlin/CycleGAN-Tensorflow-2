#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir logs \
    --name "Mura Attention Gan" \
    --description "AppleOrange Attention" \
    --one_shot
