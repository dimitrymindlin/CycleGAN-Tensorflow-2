#!/bin/bash
#SBATCH --gres=gpu:tesla
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --output=/homes/dmindlin/ABC-GAN/slurm_logs/abc-gan.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun abc-gan.job.sh