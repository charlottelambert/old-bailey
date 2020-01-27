#!/bin/bash

#SBATCH --job-name=lda
#SBATCH --output=../logs/lda_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu
#SBATCH -p emc

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export JAVA_OPTIONS="-Xms4G -Xmx8G"
export MALLET_PATH=~/Mallet/bin/mallet

./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-dh-1774 --num_topics=20

echo Finished: `date`
