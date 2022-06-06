#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/attention_5_5_1_0_mura.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun attention_5_5_1_0_mura.job.sh