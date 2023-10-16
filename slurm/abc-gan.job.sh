#!/bin/bash
source /homes/dmindlin/.bashrc
conda activate abc-gan
cd /homes/dmindlin/ABC-GAN

python3 -m train \
  --dataset mura \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 0 \
  --clf_name inception \
  --start_attention_epoch 0 \
  --discriminator attention \
  --cyclegan_mode abc-gan