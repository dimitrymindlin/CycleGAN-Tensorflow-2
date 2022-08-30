#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2

python3 -m test_counterfactuals --dataset mura --counterfactuals ganterfactual --save_img True --tcv_os True --ssim_psnr True --kid True

