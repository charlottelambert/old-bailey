## Senior Thesis Project

The following descriptions relate to code in the `thesis/code` directory. To get more information, you can run each file with the `--help` flag and see all possible arguments.

### Calculating Corpus Statistics

In order to calculate some valuable statistics about a corpus, first [download an Elementary Latin Dictionary](http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus%3Atext%3A1999.04.0060). Then, to convert the downloaded XML file to a text file, run the following command:

```
./make_latin_dict.py [PATH_TO_XML_FILE] > latin_dict.txt

```

From the XML files of BNC data gotten [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554#), extract all words and compile a list of words that can be considered modern English words. Pass in the output file (path provided by `--save_lexicon_path` option) as the `--english_words` argument to `calc_stats.py`.

```
./build_bnc.py [XML_BASE_DIR]
```

Then, run the following command to calculate useful statistics on how many modern english words, historical english words, latin words, and proper nouns are present in all the files in a specific corpus directory:

```
./calc_stats.py --corpus_dir=PATH_TO_CORPUS --latin_dict=latin_dict.txt

```

This will first run TF-IDF on the data (if no `--tfidf_model_dir_path` argument is input) and save the output model to a timestamped directory in `/work/clambert/models`. Then, this code will compute statistics and output a tsv file in a directory at `PATH_TO_CORPUS/../stats_dir` labeled to indicate over which corpus it ran.

Note, you can also supply a document to serve as an English dictionary (e.g., `/work/clambert/thesis-data/english_dict.txt`). By default, `words.words()` from `nltk` will be used. The most updated version can be obtained by running `/home/clambert/english-words/word_fix.py`.

### Topic Model

You can run LDA or a dynamic topic model using `run-model`. By default, running `./run-model` will run LDA, but you can run `./run-model dtm` to specify a dynamic topic model. Parameters for the models are given some default values (in `./run_model.py`) and additional options are passed in through `./run-lda.sbatch` and `./run-dtm.sbatch`.

- All relevant model files will be stored in either `/work/clambert/models/mallet` or `work/clambert/models/dtm` within date and time-stamped directories.
- note: running gensim's dynamic topic model still has bugs. Use [Derek Greene's](https://github.com/charlottelambert/dynamic-nmf) implementation for less buggy code.

### Word2Vec Model

To run a Word2Vec model, use `train_embedding_model.py`. This will output relevant files to `/work/clambert/models` in a time-stamped directory. You can filter out the top `n` words if you wish to visualize data using projector.tensorflow.org.

Before visualizing data, you must run the following line of code using the path to the saved model in order to output necessary files for projector.tensorflow.org:
```
./dump-w2v.py MODEL_PATH
```
This code will generate two tsv files in the same directory as the input model: `model.tsv` and `labels.tsv`. These two files can then be used to load data for visualization in projector.tensorflow.org.


### FastText MODEL

To run a FastText model, use the same process described in the above section for Word2Vec and include the flag `-f`.
