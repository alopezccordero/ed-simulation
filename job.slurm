#!/bin/bash
#SBATCH --job-name=ed-simulation-ppo
#SBATCH --partition=gpua30q
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=ed-simulation-ppo_%j.out
#SBATCH --error=ed-simulation-ppo_%j.err

source ~/miniconda/bin/activate
conda activate ed-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/ed-simulation

echo "Running on $(hostname)"
echo "current directory: $(pwd)"
python --version

python train_ppo.py