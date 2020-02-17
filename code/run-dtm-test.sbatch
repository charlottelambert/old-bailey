#!/bin/bash

#SBATCH --job-name=dtm-test
#SBATCH --output=../logs/dtm_test_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu
#SBATCH -p emc

ITERS=${1:-20}
TOPICS=${2:-20}
SEED=${3:-0}
SUFFIX=${4:-}

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export DTM_PATH=/home/clambert/dtm/dtm/main

./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-dh --num_topics=$TOPICS --num_iterations=$ITERS --seed=$SEED --model_type=dtm --year_split=100 --suffix=$SUFFIX


echo Finished: `date`
