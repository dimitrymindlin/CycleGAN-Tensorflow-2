#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

tensorboard dev upload --logdir logs/horse2zebra \
    --name "Horse2Zebra Attention" \
    --description "Horse2Zebra Attention" \
    --one_shot
