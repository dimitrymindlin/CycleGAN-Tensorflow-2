#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m train --dataset mura --generator resnet --attention_type attention-gan --gradient_penalty_weight 10 --cycle_loss_weight 1  --attention_gan_original True