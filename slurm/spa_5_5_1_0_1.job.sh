#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_spa_gan \
  --dataset horse2zebra \
  --adversarial_loss_weight 5 \
  --cycle_loss_weight 5 \
  --counterfactual_loss_weight 1 \
  --feature_map_loss_weight 0 \
  --identity_loss_weight 1 \
  --attention clf \
  --generator resnet \

