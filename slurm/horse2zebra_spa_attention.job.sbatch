#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=70:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/horse2zebra_spa_attention.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun horse2zebra_spa_attention.job.sh