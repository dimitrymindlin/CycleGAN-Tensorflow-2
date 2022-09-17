#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m experiment \
  --dataset mura \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 1 \
  --generator resnet \
  --clf_ckp_name 2022-06-04--00.05 \
  --discriminator patch_gan_attention \
  --start_attention_epoch 10 \
  --load_checkpoint 2022-09-14--13.30 \

