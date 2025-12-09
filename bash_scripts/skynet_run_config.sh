#! /bin/bash
#SBATCH -o slurm/alignment_output_%j.txt
#SBATCH -e slurm/alignment_error_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J dv2-train
#SBATCH -p rail-lab

echo "Running config: $1"

/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./train.py \
    --config $1

echo "Training completed for: $1"

