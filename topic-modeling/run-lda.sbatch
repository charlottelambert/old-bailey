#!/bin/bash

#SBATCH --job-name=lda
#SBATCH --output=../logs/lda_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu
#SBATCH -p emc --exclusive

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export JAVA_OPTIONS="-Xms4G -Xmx8G"
# export MALLET_PATH=~/Mallet/bin/mallet
export MALLET_PATH=lda-tools/ext/mallet/bin/mallet
export PYTHONPATH=/home/clambert/thesis/topic-modeling/lda-tools/lib:$PYTHONPATH

TOPICS=${1:-30}
SUFFIX=${2:-}
echo Suffix: $SUFFIX

# ./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/dtm-test --num_topics=$TOPICS --year_split=-1 --coherence --num_iterations=5 --suffix=$SUFFIX
# unigrams:
#./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

# bigrams:
# ./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

# topical n-grams
#./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-tng --num_topics=$TOPICS --year_split=-1 --topical_n_grams --suffix=$SUFFIX


# OB + LL:
# ./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
# ./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

echo Finished: `date`
