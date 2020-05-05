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


## Analyzing Topic Models

### Aligning Manual Dynamic LDA Models

For a manual dynamic LDA model run using the Mallet wrapper code in `lda-tools`, the resulting models will all have the same number of topics, but will not be matched up with corresponding topics from the other time slices' models. To align the topics found by three LDA models, run `align_topics.py`. Use the following command:

```
./align_topics WEIGHTED_KEYS_1 WEIGHTED_KEYS_2 WEIGHTED_KEYS_3
```
At this point, three models must be input. the `WEIGHTED_KEYS` arguments refer to files saved within each subdirectory within the time-stamped directory created when running `run-model`. These files are called `weighed-keys.txt`. This command will print out `NUM_TOPICS` alignments in order of most similar to least similar. After all alignments of 2 or three topics are printed, all the topics that were not aligned are printed out. These are considered unique to their own time slice. 

### Calculating Topic Coherence

To compute the topic coherence of a model, use the following command:
```
./coherence.py WEIGHTED_KEYS [...] --method=vectors --word2vec_model=PATH_TO_W2V
```

This command uses the method of computing topic coherence by using word similarities found by the Word2Vec model at `PATH_TO_W2V`. Make sure that this Word2Vec model has the same vocabulary as the model used to run the model(s) input to this file. See `../vector-space/README.md` for instructions on running a Word2Vec model over a corpus file generated by the Mallet wrapper code. Notice that you can input at least one weighted keys file and the program will compute and print out the topic coherence of each model.

The other possible coherence calculation method is `umass` which does not require that a Word2Vec model be input. The vector method is recommended, however.

### Loading Models (Gensim models only)

To load a Gensim model and print out the model's topics, run the following command:

```
./load_model.py MODEL_PATH --num_topics=-1 --model_type=lda
```

The `--num_topics` argument specifies the number of topics to print out. When the argument is `-1`, it will print out all the topics found by the model. 

The `--model_type` argument requires that the input be either `lda` (default), `dtm`, or `ldaseq` depending on how the original input model was run.

### Visualizing Models

The following commands take in model/weighted keys files and generate HTML for visualizing these topics. The commands return the HTML itself and can be redirected to an HTML file. The resulting visualization shows all the topics with the highest-weighted words in dark grey/black and the lower-weighted words in increasingly light grey.

#### Gensim

```
./vis-topic-gensim.py WEIGHTED_KEYS > vis_gensim.html
```

#### Mallet Wrapper

```
./vis_topic_mallet.py WEIGHTED_KEYS > vis_mallet.html
```

## ETM/D-ETM (need to update)
Code for running ETM on Old Bailey data can be cloned from [this](https://github.com/charlottelambert/ETM) repository.
Code for running D-ETM Old Bailey data can be cloned from [this](https://github.com/charlottelambert/DETM) repository.
