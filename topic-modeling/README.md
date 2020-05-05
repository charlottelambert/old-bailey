# thesis/topic-modeling

The following table shows all files present in the `thesis/data` directory. Below the table there are descriptions of how to run each file.

Name | Content
-------|-------
`align_topics.py` | Code to align topics from LDA models that are split into different time slices
`coherence.py` | Code to calulate topic coherence for a given model
`load_model.py` | Code to load and print topic model topics
`README.md` | This file
`run-dtm.sbatch` | File to run DTM using slurm (called in run-model)
`run-dtm-test.sbatch` | File to run several DTM models using slurm (called in run-model)
`run-lda.sbatch` | File to run LDA using slurm (called in run-model)
`run-model` | Bash script to run one or more topic models using slurm
`run_model.py` | Code for running topic models (called by `.sbatch` scripts)
`vis-topic-gensim.py` | Code to generate html visualization of LDA model run using Gensim
`vis_topic_mallet.py` | Code to generate html visualization of LDA model run using wrapper code in `lda-tools`


## LDA/DTM (need to update)

You can run LDA or a dynamic topic model using `run-model`. By default, running `./run-model` will run LDA, but you can run `./run-model dtm` to specify a dynamic topic model. Parameters for the models are given some default values (in `./run_model.py`) and additional options are passed in through `./run-lda.sbatch` and `./run-dtm.sbatch`.

- All relevant model files will be stored in either `/work/clambert/models/mallet` or `work/clambert/models/dtm` within date and time-stamped directories.
- note: running gensim's dynamic topic model still has bugs. Use [Derek Greene's](https://github.com/charlottelambert/dynamic-nmf) implementation for less buggy code.

## ETM/D-ETM (need to update)
Code for running ETM on Old Bailey data can be cloned from [this](https://github.com/charlottelambert/ETM) repository.
Code for running D-ETM Old Bailey data can be cloned from [this](https://github.com/charlottelambert/DETM) repository.
