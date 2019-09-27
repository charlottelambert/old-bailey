#!/bin/bash

#SBATCH -c 64
#SBATCH --job-name=lda
#SBATCH --output=../logs/lda_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export MALLET_PATH=~/Mallet/bin/mallet
export JAVA_OPTIONS="-Xms4G -Xmx8G"

./run_lda.py --unigrams_only

echo Finished: `date`
