#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --output=/media/compute/homes/dmindlin/CycleGAN-Tensorflow-2/slurm_logs/attention_1_10_0_1_199_a2o.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun attention_1_10_0_1_199_a2o.job.sh