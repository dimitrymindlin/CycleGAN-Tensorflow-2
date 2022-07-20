#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_attention_gan \
  --dataset horse2zebra \
  --discriminator_loss_weight 5 \
  --cycle_loss_weight 5 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 0 \
  --generator resnet \
  --start_attention_epoch 80 \

