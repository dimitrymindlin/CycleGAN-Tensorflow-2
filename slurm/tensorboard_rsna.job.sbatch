#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=2:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/tensorboard_rsna.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1

srun tensorboard_rsna.job.sh