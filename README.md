## Senior Thesis Project

The following descriptions relate to code in the `thesis/code` directory. To get more information, you can run each file with the `--help` flag and see all possible arguments.

### Corpus Pre-Processing

Convert given XML documents to text files using the following code:
```
./data_reader /work/clambert/thesis-data/sessionsPapers

```

Converted text data will be placed in a directory with the suffix `-txt` added to the input `corpus_dir`. By default, code will not overwrite files if they already exist in this created directory. Flags can be added to change this and to incorporate annotations from the XML into the text data.

### Tokenization

To tokenize data (and clean up some idiosyncrasies), run `run_tokenize.py` on a corpus of text files. Some default arguments have been added for simplicity, but can be overridden by setting the appropriate flags.

```
./run_tokenize.py --corpus_dir=CORPUS_DIR
```

Tokenized text files will be written to a directory with the suffix `-tok` added to the input `CORPUS_DIR`. Like with `code/data_reader.py`, by default, code will not overwrite files if they already exist in this created directory. Flags can be added to change this and to incorporate annotations from the XML into the text data.

### Dehyphenation

To dehyphenate data, run `dehyphenate.py` on a single text file. Typically, dehyphenation should only be done after data has been tokenized. You can run the code as follows, using parallelization for faster processing.

```
srun -c 64 --pty /bin/bash
ls -d CORPUS_DIR/* | parallel --progress -j 64 "./dehyphenate.py {}"
```

This command will dehyphenate all files in `CORPUS_DIR` and output all files into `CORPUS_DIR-dh`.

### Bigrams (optional)

To convert unigram data to bigram data, run the following line of code:

```
./run_tokenize.py --corpus_dir=CORPUS_DIR --bigrams
```
This command will convert all files in `CORPUS_DIR` to a bigram representation and output all files into `CORPUS_DIR-bi`.

Note: the data provided should be tokenized and dehyphenated first. If data is not already dehyphenated, there is no code to properly dehyphenate bigrams.


### Calculating Corpus Statistics

In order to calculate some valuable statistics about a corpus, first [download an Elementary Latin Dictionary](http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus%3Atext%3A1999.04.0060). Then, to convert the downloaded XML file to a text file, run the following command:

```
./make_latin_dict.py [PATH_TO_XML_FILE] > latin_dict.txt

```

Then, run the following command to calculate useful statistics on how many modern english words, historical english words, latin words, and proper nouns are present in all the files in a specific corpus directory:

```
./calc_stats.py --corpus_dir=PATH_TO_CORPUS --latin_dict=latin_dict.txt

```

This will output a tsv file in a directory at `PATH_TO_CORPUS/../stats_dir` labeled to indicate over which corpus it ran.

### Topic Model

You can run LDA or a dynamic topic model using `run-model`. By default, runnign `./run-model` will run LDA, but you can run `./run-model dtm` to specify a dynamic topic model. Parameters for the models are given some default values (in `./run_model.py`) and additional options are passed in through `./run-lda.sbatch` and `./run-dtm.sbatch`.

- All relevant model files will be stored in either `./models/mallet` or `./models/dtm` within date and time-stamped directories.
- note: running gensim's dynamic topic model still has bugs. Use [Derek Greene's](https://github.com/charlottelambert/dynamic-nmf) implementation for less buggy code.
