#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m test_counterfactuals \
--dataset apple2orange \
--tcv_os True \
--ssim_psnr True \
--kid True \
--save_img False \
--clf_name inception \
--generator resnet \
--attention_type none


