#!/bin/bash

#SBATCH --job-name=dtm
#SBATCH --output=../logs/dtm_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clambert@vassar.edu
#SBATCH -p emc

echo Running: $SLURM_ARRAY_TASK_ID
uname -a
date

export DTM_PATH=/home/clambert/dtm/dtm/main

./run_model.py --unigrams_only --corpus_dir=/work/clambert/thesis-data/dtm-test --num_topics=20 --model_type=dtm

echo Finished: `date`
