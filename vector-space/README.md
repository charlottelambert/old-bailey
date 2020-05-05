# thesis/vector-space

The following table shows all files present in the `thesis/vector-space` directory. Below the table there are descriptions of how to run each file.

Name | Contents
-------|-------
`dump_w2v.py` | Code called in `train_embedding_model.py` for writing `tsv` and `txt` data used by projector.tensorflow.org
`README.md` | This file
`train_embedding_model.py` | File for running Word2Vec and FastText models

## Word2Vec and FastText Models

To run a Word2Vec model, use `train_embedding_model.py`. Run one of the following commands, for example, or specify other parameters. All of the below commands will output relevant files to a time-stamped directory within the argument passed to `--save_model_dir`.

Below are some parameters and instructions of when to use them. Use the `--help` flag to see all possible arguments.
Argument | Description
---------|------------
`pretrained` | Path to pretrained word embeddings. If specified, loaded model will be retrained using the vocabular of the input corpus. This increases the vocabulary.
`filter_top_words` | Model will filter out the top `n` words of the model. Use this with `n=10000` when visualizing model with (projector.tensorflow.org)[projector.tensorflow.org]. Otherwise, specify `n=0`.
`find_n_neighbors` | Number of neighbors to find for each word in a list of relevant words (`word_list` in `train_embedding_model.py`). Program will output TSV file containing the `n` nearest neighbors and generate wordcloud visualizations of these word-relationships.
`f` | Whether or not to train a FastText model instead of Word2Vec

### Word2Vec from Corpus Directory
```
./train_embedding_model.py --corpus_dir=CORPUS_DIR
````

This runs Word2Vec over all the text files in `CORPUS_DIR`.

### Word2Vec from Corpus TSV File
```
./train_embedding_model.py --corpus_file=CORPUS_FILE
````

This command will treat each line in the input `CORPUS_FILE` as a document and run Word2Vec over all documents in the file.

### Word2Vec from Corpus txt File (output from Mallet wrapper)

If you want to compute topic coherence over a model run using the Mallet wrapper code in `../topic-modeling/lda-tools`, you first need to train a Word2Vec model using the `corpus.txt` file written by the wrapper code before training. This ensures that the Word2Vec vocabulary is identical to that of the model. Use the following arguments:

```
./train-embedding_model.py --corpus_txt_file=PATH_TO_CORPUS_FILE --filter_top_words=0 --year_split=-1
```
Make sure not to filter out the top 10,000 words because this will cause the vocabularies to not match. You can use the `--year_split` argument to match the same parameter from the command used to run `run_model.py`. Alternatively, always input `--year_split=-1` so the output Word2Vec model can be used for calculating topic coherence for a model run with no year split and for models run with a year split.



## Files for projector.tensorflow.org

When running `train_embeddings.py`, all the files needed to visualize on (projector.tensorflow.org)[projector.tensorflow.org] are output to the same directory as the saved model. You can also generate these relevant files separately using `dump_w2v.py`

```
./dump-w2v.py MODEL_PATH
```

This code will generate two tsv files in the same directory as the input model: `model.tsv` and `labels.tsv`. These two files can then be used to load data for visualization in projector.tensorflow.org.


