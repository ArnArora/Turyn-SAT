#!/bin/bash 
#SBATCH --account=def-vganesh
#SBATCH --mail-user=aarora362@gatech.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-00:10
#SBATCH --output=training-run-7-%j.out
module load StdEnv/2023 python/3.10 cuda/12.2

virtualenv --no-download $SLURM_TMPDIR/env 
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip
pip install torch numpy python-sat

python turyn_policy_transformer.py 7 --test-frac 0.2 --epochs 10