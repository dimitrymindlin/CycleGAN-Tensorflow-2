#!/bin/bash
#SBATCH --gres=gpu:tesla
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/abc-gan.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun abc-gan.job.sh