#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_spa_gan \
  --dataset horse2zebra \
  --generator resnet \
  --feature_map_loss_weight 0 \
  --attention_intensity 0


