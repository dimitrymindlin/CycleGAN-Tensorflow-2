#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_mura \
  --dataset mura \
  --discriminator_loss_weight 1 \
  --cycle_loss_weight 1 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 1