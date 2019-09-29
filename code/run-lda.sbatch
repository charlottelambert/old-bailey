#!/bin/bash

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

./run_lda.py --unigrams_only --corpus_dir=../data/sessionsPapers-txt-tok

echo Finished: `date`
