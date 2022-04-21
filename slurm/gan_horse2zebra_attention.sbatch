#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/gan_horse2zebra_attention.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun gan_horse2zebra_attention.job.sh