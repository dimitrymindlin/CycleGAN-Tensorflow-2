#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/evaluation/scripts/

python3 -m evaluate_robustness \
--dataset mura \


