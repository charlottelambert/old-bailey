#!/bin/bash

#SBATCH --job-name=dtm
#SBATCH --output=../logs/dtm_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu
#SBATCH -p emc

MODEL_TYPE=${1:-dtm}

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export MALLET_PATH=lda-tools/ext/mallet/bin/mallet
export PYTHONPATH=/home/clambert/thesis/code/lda-tools/lib:$PYTHONPATH
export DTM_PATH=/home/clambert/dtm/dtm/main

# ./run_model.py --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-bi-tng --model_type=$MODEL_TYPE --year_split=100 --vis --num_topics=30 --num_iterations=1000
./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok --model_type=$MODEL_TYPE --year_split=100 --vis --num_topics=30 --num_iterations=1000
echo Finished: `date`
