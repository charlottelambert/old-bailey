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
export MALLET_PATH=lda-tools/ext/mallet/bin/mallet
export PYTHONPATH=/home/clambert/thesis/topic-modeling/lda-tools/lib:$PYTHONPATH

TOPICS=${1:-30}
SUFFIX=${2:-}
echo Suffix: $SUFFIX

# DATA FROM TSV FILES
# Old Bailey bigram data only, year split 100
#./run_model.py --unigrams_only --corpus_file=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-bi.tsv --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

# Old Bailey and London Lives bigram data, year split 100
#./run_model.py --unigrams_only --corpus_file=/work/clambert/thesis-data/OB_LL-txt-tok-bi.tsv --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

#######################################################################################################

# DATA FROM DIRECTORIES
# Old Bailey data only
# unigrams:
#./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-min_df_5-max_df_0.8 --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

# bigrams:
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-min_df_5-max_df_0.8 --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

# topical n-grams
#./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-tng --num_topics=$TOPICS --year_split=-1 --topical_n_grams --suffix=$SUFFIX

# Old Bailey and London Lives data: time slices, bigrams
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok --num_topics=$TOPICS --year_split=-1 --suffix=$SUFFIX
# time slices
#./run_model.py --bigrams_only --corpus_dir=/work/clambert/thesis-data/OB_LL-txt-tok-min_df_10-max_df_0.8 --num_topics=$TOPICS --year_split=100 --suffix=$SUFFIX

echo Finished: `date`
