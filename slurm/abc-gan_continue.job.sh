#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m experiment \
  --dataset rsna \
  --adversarial_loss_weight 1 \
  --cycle_loss_weight 10 \
  --counterfactual_loss_weight 1 \
  --identity_loss_weight 0 \
  --generator resnet \
  --epochs 22 \
  --clf_name alexnet \
  --start_attention_epoch 10 \
  --discriminator patch_gan_attention \
  --load_checkpoint "2023-01-05--09.51"

