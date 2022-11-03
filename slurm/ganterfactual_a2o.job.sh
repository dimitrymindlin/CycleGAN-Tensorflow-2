#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m experiment \
  --dataset apple2orange \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 5 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 1 \
  --generator resnet \
  --epoch_decay 100 \
  --start_attention_epoch 199 \
  --discriminator patch-gan \
  --attention_type none

