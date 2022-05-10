#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train \
  --dataset horse2zebra \
  --generator resnet \
  --attention_type attention-gan-original \
  --discriminator_loss_weight 10 \
  --cycle_loss_weight 1 \
  --counterfactual_loss_weight 0
