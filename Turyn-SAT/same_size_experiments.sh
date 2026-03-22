#!/bin/bash 
#SBATCH --account=def-vganesh
#SBATCH --mail-user=aarora362@gatech.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-10:00
#SBATCH --output=training-run-11-%j.out
module load StdEnv/2023 python/3.10 cuda/12.2

virtualenv --no-download $SLURM_TMPDIR/env 
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip
pip install torch numpy python-sat

python turyn_policy_transformer.py 11 --test-frac 0.2 --epochs 20 --partial-flat-len 4 --limit 20000 --save-model model_11.pt --save-data data_11.pkl