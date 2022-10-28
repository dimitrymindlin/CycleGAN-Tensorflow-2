#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/ganterfactual_rsna_resnet.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun ganterfactual_rsna_resnet.job.sh