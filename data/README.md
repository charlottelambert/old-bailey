# thesis/data

The following table shows all files present in the `thesis/data` directory. Below the table there are descriptions of how to run each file.

Name | Contents
-------|-------
`build_bnc.py` | 
`custom_stop_words.py` | Custom list of stop words to add to NLTK's list
`data_reader.py` | File to convert XML data to text data
`make_latin_dict.py` | Builds a Latin dictionary
`ngrams.py` | File to write unigram and bigram personal word lists
`parallel-tokenize` | Bash script to run `run_tokenize.py` in parallel on an input tsv file
`prep_tsv.py` | File to split or merge tsv file(s) to allow for parallel processing of one tsv data file
`README.md` | This file
`run_tokenize.py` | File to tokenize data

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


## Tokenization

To tokenize data (and clean up some idiosyncrasies), run `run_tokenize.py` on a corpus of text files or one tsv file. Some default arguments have been added for simplicity, but can be overridden by setting the appropriate flags.

```
./run_tokenize.py --tsv_data=sessionsAndOrdinarys-txt.tsv --overwrite
```

If tsv file is input, output tokenization will be written to a tsv file with the suffix `-tok`. Additional arguments may add to this suffix. If no tsv file is input but something is passed into the argument `--corpus_dir`, each file in the directory will be tokenized and written to a file of the same name in a directory of the form `corpus_dir-tok`. You can also provide a path to a specific file using the option `--filepath` to tokenize that file only. It will be processed the same as any file input using the `--corpus_dir` argument. This allows for parallelization.

If you wish to disable spell-checking (a function that will split words that appear to be merged based on whether or not it is present in the input word lists or a British MySpell dictionary), include the flag `--disable_spell_check`.

Use the `--help` flag to get mroe information about remaining flags and arguments.

### TSV Files in Parallel (need to update)

`prep_tsv.py` and `parallel-tokenize`

### Bigrams (optional)

To convert unigram data to bigram data, run the following line of code, run `run_tokenize.py` with the flag `--bigrams`. Output data will be in bigram representation.

