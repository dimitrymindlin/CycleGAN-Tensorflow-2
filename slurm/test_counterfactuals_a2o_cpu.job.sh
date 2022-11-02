#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m test_counterfactuals \
--dataset apple2orange \
--tcv_os False \
--ssim_psnr False \
--kid False \
--save_img True \
--clf_name inception \
--generator resnet

