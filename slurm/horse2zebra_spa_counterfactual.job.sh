#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_spa_gan \
  --dataset horse2zebra \
  --generator resnet \
  --counterfactual_loss_weight 1 \
  --attention clf \
  --load_checkpoint 2022-05-18--11.52


