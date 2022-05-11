#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train \
  --dataset horse2zebra \
  --generator unet \
  --attention_type none \
  --discriminator_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 0
