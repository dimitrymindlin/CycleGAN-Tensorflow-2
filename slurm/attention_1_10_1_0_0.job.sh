#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m experiment \
  --dataset mura \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 0 \
  --clf_ckp_name 2022-06-04--00.05 \
  --generator resnet \
  --clf_name inception \
  --start_attention_epoch 0 \
  --discriminator patch_gan_attention

