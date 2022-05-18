#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train \
  --dataset horse2zebra \
  --load_checkpoint 2022-05-16--21.40 \
  --feature_map_loss_weight 0 \
  --attention_type none \
  --generator resnet


