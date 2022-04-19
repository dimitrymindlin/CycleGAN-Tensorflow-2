#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/gan_apple_orange_gradcam_plus.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun gan_apple_orange_gradcam_plus.job.sh