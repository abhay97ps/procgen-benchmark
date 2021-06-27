#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --mem=16384
#SBATCH --job-name=test
#SBATCH --mail-user=aps647@nyu.edu
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda10.1/2.6.4
source pytorch_env/py3.6.3/bin/activate

cd RL
python train.py --exp_name attention_25m --env_name chaser --start_level 0 --num_levels 200 --param_name easy-custom --device gpu --log_level 30 --num_timesteps 25000000