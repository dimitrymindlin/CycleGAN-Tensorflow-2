#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train_attention_gan \
  --dataset apple2orange \
  --discriminator_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 0 \
  --identity_loss_weight 5 \
  --generator resnet \
  --start_attention_epoch 50 \

