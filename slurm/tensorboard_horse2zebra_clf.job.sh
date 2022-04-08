#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir logs/horse2zebra_clf \
    --name "Horse 2 Zebra CLF" \
    --description "Horse 2 Zebra CLF" \
    --one_shot
