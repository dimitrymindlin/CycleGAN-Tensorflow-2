#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train \
  --dataset horse2zebra \
  --generator resnet \
  --epochs 100 \
  --epoch_decay 50 \
  --attention_type none \
  --attention_gan_original False \
  --discriminator_loss_weight 10 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 0
