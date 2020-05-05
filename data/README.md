# thesis/data

The following table shows all files present in the `thesis/data` directory. Below the table there are descriptions of how to run each file.

Name | Contents
-------|-------
`build_bnc.py` | 
`custom_stop_words.py` | Custom list of stop words to add to NLTK's list
`data_reader.py` | File to convert XML data to text data
`make_latin_dict.py` | Builds a Latin dictionary
`ngrams.py` | File to write unigram and bigram personal word lists
prep_tsv.py | File to split or merge tsv file(s) to allow for parallel processing of one tsv data file
README.md | This file
run_tokenize.py | File to tokenize data

## Collection of Data

Convert given XML documents to text files using the following code:
```
./data_reader /work/clambert/thesis-data/sessionsPapers --tsv=1 --overwrite
```

If the `--tsv` flag is passed in as true, converted data will be output to a tsv file with the suffix `-txt.tsv` added to the input `corpus_XML_dir` with one line per document. Otherwise, converted text data will be placed in a directory with the suffix `-txt` added to the input `corpus_XML_dir`. The `--overwrite` flag indicates that if the output tsv file or directory already exists, it should be overwritten.

Additionally, if the data passed in is from the London Lives corpus, include the flag `--london_lives` to ensure the data is collected properly.

Finally, up to one of two flags can be passed in to indicate that annotations from the input XML should be replaced with some token. The `--encode_annotations_general` flag will result in output data in which every person's name is replaced with a token of the format `speakerType_gender`. The `--encode_annotations_specific` flag will result in output data in which every person's name is replaced with a token of the format `speakerType_GIVENNAME_SURNAME`. In either case, if something is unknown, it will be replaced with the token `unk`.


## Latin dictionary (need to update)

In order to calculate some valuable statistics about a corpus, first [download an Elementary Latin Dictionary](http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus%3Atext%3A1999.04.0060). Then, to convert the downloaded XML file to a text file, run the following command:

```
./make_latin_dict.py [PATH_TO_XML_FILE] > latin_dict.txt

```

From the XML files of BNC data gotten [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554#), extract all words and compile a list of words that can be considered modern English words. Pass in the output file (path provided by `--save_lexicon_path` option) as the `--english_words` argument to `calc_stats.py`.

```
./build_bnc.py [XML_BASE_DIR]
```

Then, run the following command to calculate useful statistics on how many modern english words, historical english words, latin words, and proper nouns are present in all the files in a specific corpus directory:


## Tokenization (need to update)

To tokenize data (and clean up some idiosyncrasies), run `run_tokenize.py` on a corpus of text files. Some default arguments have been added for simplicity, but can be overridden by setting the appropriate flags.

```
./run_tokenize.py --corpus_dir=CORPUS_DIR
```

Tokenized text files will be written to a directory with the suffix `-tok` added to the input `CORPUS_DIR`. Like with `code/data_reader.py`, by default, code will not overwrite files if they already exist in this created directory. Flags can be added to change this and to incorporate annotations from the XML into the text data.

You can also provide a path to a specific file using the option `--filepath` to tokenize that file only. This allows for parallelization.

### Bigrams (optional)

To convert unigram data to bigram data, run the following line of code:

```
./run_tokenize.py --corpus_dir=CORPUS_DIR --bigrams
```
This command will convert all files in `CORPUS_DIR` to a bigram representation and output all files into `CORPUS_DIR-bi`.

Note: the data provided should be tokenized and dehyphenated first. If data is not already dehyphenated, there is no code to properly dehyphenate bigrams.
