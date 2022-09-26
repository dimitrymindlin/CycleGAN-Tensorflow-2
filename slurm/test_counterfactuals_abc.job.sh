#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m test_counterfactuals \
--dataset horse2zebra \
--counterfactuals_type abc-gan \
--tcv_os True \
--ssim_psnr True \
--kid True \
--save_img True \
--clf_ckp_name 2022-06-04--00.00 \


