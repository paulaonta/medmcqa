#!/bin/bash
#SBATCH --job-name=TRAIN_BERT
#SBATCH	--cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH	--gres=gpu:1
#SBATCH --output=./output.log
#SBATCH --error=./errore.err

source ../../transformers-3.1.0/bin/activate

srun pip install dataclasses

srun python train.py --model bert-base-uncased
