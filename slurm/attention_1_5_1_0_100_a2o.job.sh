#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m experiment \
  --dataset apple2orange \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 5 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 0 \
  --generator resnet \
  --clf_ckp_name 2022-09-23--15.18 \
  --discriminator patch_gan \
  --epoch_decay 150 \
  --start_attention_epoch 100 \

