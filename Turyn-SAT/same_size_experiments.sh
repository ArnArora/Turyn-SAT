#!/bin/bash 
#SBATCH --account=def-vganesh
#SBATCH --nodes=1 
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G 
#SBATCH --time=3-00:00
module load StdEnv/2023 python/3.10 cuda/12.2

virtualenv --no-download $SLURM_TMPDIR/env 
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -r requirements.txt

python turyn_policy_transformer.py 7 --test-frac 0.2 --epochs 10