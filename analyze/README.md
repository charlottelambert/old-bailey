# Code to analyze data

Name | Content
-------|-------
`analyze_utils.py` | (is this used?)
`calc_stats.py` | Code to calculate statistics on processed data
`README.md` | This file
`train_tfidf.py` | Code for training TF-IDF models

## Calculating Corpus Statistics (need to update)

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

