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


## Running LDA/DTM/LDAseq

### `run_model.py`

This file is the python code used to run LDA, manual dynamic LDA, DTM, and LDAseq. The following command is an example of running LDA over a TSV file containing the Old Bailey corpus:

```
./run_model.py --model_type=lda --corpus_file=[TSV_DATA] --num_topics=NUM_TOPICS --year_split=-1
```

This will run an LDA model over the entire corpus found in `TSV_DATA` with `NUM_TOPICS` topics. By default, this will save an LDA model (along with a file containing the parameters used to run the model) in a time-stamped directory within the path provided in the `--save_model_dir` argument. To run manual dynamic LDA, change the `--year_split` argument to the desired number of years per time slice, default is 100. This will save the individually-run models in subdirectories of the time-stamped directory with names indicating the first year in that model's time slice. 

If no file is passed into `--corpus_file`, the value for `--corpus_dir`, a directory containing the data in text files, will be used instead. If the input `--corpus_file` only contains Old Bailey data and you wish to run the model over both Old Bailey and London Lives data, pass in the path to a tsv file containing the London Lives data to the `--london_lives_file` argument in addition to the `--corpus_file` argument.

Including the flag `--gensim` indicates that you wish to run LDA with Gensim's wrapper. It is recommended that you do not use this flag and instead let the Mallet wrapper code in `lda-tools` run LDA. See the `README.md` in the `thesis` directory for instructions on obtaining the Mallet wrapper.

When running LDA on the files in a directory, you may specify the `--bigrams_only` argument to convert the data to bigrams. This will be reported to the Mallet wrapper code. 

You can also change the type of model being run to `dtm` or `ldaseq`.

Use the `--help` flag to get information about the remaining arguments.

### `run-lda.sbatch` and `run-dtm.sbatch`

These files will run `run_model.py` with appropriate parameters for LDA. You can run these files with the following commands:
```
sbatch run-lda.sbatch NUM_TOPICS
sbatch run-dtm.sbatch MODEL_TYPE
```
Modify the parameters passed into `run_model.py` within this file to change how LDA or DTM is run. The file contains several examples of running LDA/DTM on the Proceedings. note that `MODEL_TYPE` must be either `dtm` or `ldaseq`.


### `run-dtm-test.sbatch`

This script is intended to run several DTM/LDAseq models. Execute a command in the following format:
```
./run-dtm-test.sbatch ITERS NUM_TOPICS SEED SUFFIX
```
This only runs one model with the input parameters (and those that can be modified in the call to `run_model.py`), so it is recommended that you run this through `run-model` for simplicity. It will loop and run this script several times. See the below explanation for further instructions.

### `run-model`

This file is a bash script to make running LDA, DTM, and LDAseq easier. Execute the following command:

```
./run-model [LDA_TYPE]
```

In this line, `LDA_TYPE` must be one of the following options:
1. `lda`: Running regular LDA. Modify the call to `sbatch run-lda.sbatch [NUM_TOPICS]` in `run-model` with the number of desired topics. Modify `run-lda.sbatch` as described above to control the parameters of LDA.
2. `lda-test`: Running several LDA models with varying number of topics. Modify the `TOPIC_ARR` variable in `run-model` to control the number of topics running. Note that the output model directories will include a suffix `-NUM` to the timestamped directories to distinguish between each model in the test (`NUM` will be an integer between 0 and the length of `TOPIC_ARR`).
3. `dtm`: Run Gensim's DTM model. Modify the number of topics in `run-dtm.sbatch` as described above.
4. `dtm-test`: Runs several versions of Gensim's DTM. The number of iterations and topics will be varied. To change the varied values, modify `ITER_ARR` and `TOPIC_ARR` in `run-model`. Output directories will include a suffix `-NUM` to the timestamped directories to distinguish between each model in the test  (`NUM` will be an integer between 0 and the number of tests being run).
5. `ldaseq`: Run Gensim's LDAseq model. Modify the number of topics in `run-dtm.sbatch` as described above.


## ETM/D-ETM (need to update)
Code for running ETM on Old Bailey data can be cloned from [this](https://github.com/charlottelambert/ETM) repository.
Code for running D-ETM Old Bailey data can be cloned from [this](https://github.com/charlottelambert/DETM) repository.
