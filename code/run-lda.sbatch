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
# export MALLET_PATH=~/Mallet/bin/mallet
export MALLET_PATH=lda-tools/ext/mallet/bin/mallet
export PYTHONPATH=/home/clambert/thesis/code/lda-tools/lib:$PYTHONPATH

# unigrams:
# ./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=30 --year_split=-1
# time slices
#./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=30 --year_split=100

# bigrams:
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=30 --year_split=-1
# time slices
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=30 --year_split=100

# OB + LL:
./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok --num_topics=100 --year_split=-1
# time slices
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok --num_topics=100 --year_split=100

echo Finished: `date`
