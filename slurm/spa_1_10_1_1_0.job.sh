#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_spa_gan \
  --dataset horse2zebra \
  --discriminator_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 1 \
  --feature_map_loss_weight 1 \
  --identity_loss_weight 0 \
  --attention clf \
  --generator resnet-attention \
  --load_checkpoint 2022-05-26--15.53


