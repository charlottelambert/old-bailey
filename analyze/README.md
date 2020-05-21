# Code to analyze data

Name | Content
-------|-------
`analyze_utils.py` | (is this used?)
`build_bnc.py` | Build lexicon of words from the BNC from BNC XML files
`calc_stats.py` | Code to calculate statistics on processed data
`make_latin_dict.py` | Builds a Latin dictionary
`README.md` | This file
`train_tfidf.py` | Code for training TF-IDF models

## Calculating Corpus Statistics

### Compiling necessary files

In order to calculate some valuable statistics about a corpus, first [download an Elementary Latin Dictionary](http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus%3Atext%3A1999.04.0060). Then, to convert the downloaded XML file to a text file, run the following command:

```
./make_latin_dict.py [PATH_TO_XML_FILE] > latin_dict.txt
```

From the XML files of BNC data gotten [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554#), extract all words and compile a list of words that can be considered modern English words. Pass in the output file (path provided by `--save_lexicon_path` option) as the `--english_words` argument to `calc_stats.py`.

```
./build_bnc.py [XML_BASE_DIR] --save_lexicon_path=bnc_lexicon.txt
```

## Statistics

Then, run the following command to calculate useful statistics on how many modern english words, historical english words, latin words, and proper nouns are present in all the files in a specific corpus directory. Pass in the file saved by `build_bnc.py` as the `--english_words` argument to `calc_stats.py`.

```
./calc_stats.py --corpus_dir=PATH_TO_CORPUS --latin_dict=latin_dict.txt --english_words=bnc_lexicon.txt --year_split=100

```

This will first run TF-IDF on the data (if no `--tfidf_model_dir_path` argument is input) and save the output model to a timestamped directory in `/work/clambert/models`. Then, this code will compute statistics and output a tsv file in a directory at `PATH_TO_CORPUS/../stats_dir` labeled to indicate over which corpus it ran.

If you want, you can replace the `--corpus_dir` argument with the `--tsv_corpus` argument and pass in a path to the tsv file containing the corpus.

Note, you can also supply a document to serve as an English dictionary (e.g., `/work/clambert/thesis-data/english_dict.txt`). By default, `words.words()` from `nltk` will be used. The most updated version can be obtained by running `/home/clambert/english-words/word_fix.py`.

Including the `--basic_stats` flag may be more useful. It will calculate how many documents, how many tokens, and how many types there are per time slice and overall. It will also find the most common word in each time slice and overall.

## TF-IDF

The code in `calc_stats.py` will run its own tf-idf model if no path is specified, but you can also generate a tf-idf model separate from calculating statistics. Run the following command:

```
./train_tfidf.py --corpus_dir=PATH_TO_CORPUS --year_split=100
```

Like always, replace the `--corpus_dir` argument with the `--tsv_corpus` argument to run over a tsv representation of the corpus.
