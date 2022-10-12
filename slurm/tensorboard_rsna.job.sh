#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir logs/rsna \
    --name "ABC-GAN for RSNA with Inception Model " \
    --description "" \
    --one_shot