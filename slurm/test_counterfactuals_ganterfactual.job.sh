#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/test_scripts

python3 -m test_counterfactuals --dataset mura --counterfactuals ganterfactual

