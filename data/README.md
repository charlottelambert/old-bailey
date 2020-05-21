# thesis/data

The following table shows all files present in the `thesis/data` directory. Below the table there are descriptions of how to run each file.

Name | Contents
-------|-------
`custom_stop_words.py` | Custom list of stop words to add to NLTK's list
`data_reader.py` | File to convert XML data to text data
`ngrams.py` | File to write unigram and bigram personal word lists
`parallel-tokenize` | Bash script to run `run_tokenize.py` in parallel on an input tsv file
`prep_tsv.py` | File to split or merge tsv file(s) to allow for parallel processing of one tsv data file
`README.md` | This file
`run_tokenize.py` | File to tokenize data

## Collection of Data

Convert given XML documents to text files using the following code:
```
./data_reader /work/clambert/thesis-data/sessionsAndOrdinarys --tsv=1 --overwrite --split_trials
```

If the `--tsv` flag is passed in as true (1), converted data will be output to a tsv file with the suffix `-txt.tsv` added to the input `corpus_XML_dir` with one line per document. Otherwise, converted text data will be placed in a directory with the suffix `-txt` added to the input `corpus_XML_dir`. The `--overwrite` flag indicates that if the output tsv file or directory already exists, it should be overwritten. The `--split_trials` flag indicates whether or not the data will be split by trial when possible. If false, output will be split by document (i.e., one session). Ordinarys Accounts are always written to one line in a tsv document or one file in the output directory since there are no trials.

Additionally, if the data passed in is from the London Lives corpus, include the flag `--london_lives` to ensure the data is collected properly. London lives data cannot be split by trial (there are no trials).

Finally, up to one of two flags can be passed in to indicate that annotations from the input XML should be replaced with some token. The `--encode_annotations_general` flag will result in output data in which every person's name is replaced with a token of the format `speakerType_gender`. The `--encode_annotations_specific` flag will result in output data in which every person's name is replaced with a token of the format `speakerType_GIVENNAME_SURNAME`. In either case, if something is unknown, it will be replaced with the token `unk`.


## Latin dictionary

In order to calculate some valuable statistics about a corpus, first [download an Elementary Latin Dictionary](http://www.perseus.tufts.edu/hopper/dltext?doc=Perseus%3Atext%3A1999.04.0060). Then, to convert the downloaded XML file to a text file, run the following command:

```
./make_latin_dict.py [PATH_TO_XML_FILE] > latin_dict.txt
```

From the XML files of BNC data gotten [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554#), extract all words and compile a list of words that can be considered modern English words. Pass in the output file (path provided by `--save_lexicon_path` option) as the `--english_words` argument to `calc_stats.py`.

```
./build_bnc.py [XML_BASE_DIR]
```

Then, run the following command to calculate useful statistics on how many modern english words, historical english words, latin words, and proper nouns are present in all the files in a specific corpus directory:

## Building personal word list (PWL) and bigram dictionary

In the next step, tokenization, some words merged by the transcription will be split. Build a dictionary of all bigrams in the corpus and a list of all unigrams in the corpus to provide the next step with more information about what words are present in the corpus. Essentially allows you to use words unique to this corpus in the process of spell checking.

```
./ngrams.py --corpus_dir=sessionsAndOrdinarys-txt --overwrite
```

This command will write bigram and unigram counts to files within the `--corpus_dir` as well as a text file serving as the unigram personal word list. The `--overwrite` flag will write to the files even if they exist. To run with tsv input, replace the `--corpus_dir` argument with `--tsv_corpus` and pass in the path to a tsv file containing the corpus. Include the `--disable_filter` flag to include all data in counts and word lists. Otherwise, only files that were manually transcribed (within year range 1674-1834) will be included.

It is recommended to run this code on the combination of London Lives data and Old Bailey data.

## Tokenization

To tokenize data (and clean up some idiosyncrasies), run `run_tokenize.py` on a corpus of text files or one tsv file. Some default arguments have been added for simplicity, but can be overridden by setting the appropriate flags.

```
./run_tokenize.py --tsv_corpus=sessionsAndOrdinarys-txt.tsv --overwrite
```

If tsv file is input, output tokenization will be written to a tsv file with the suffix `-tok`. Additional arguments may add to this suffix. If no tsv file is input but something is passed into the argument `--corpus_dir`, each file in the directory will be tokenized and written to a file of the same name in a directory of the form `corpus_dir-tok`. You can also provide a path to a specific file using the option `--filepath` to tokenize that file only. It will be processed the same as any file input using the `--corpus_dir` argument. This allows for parallelization.

If you wish to disable spell-checking (a function that will split words that appear to be merged based on whether or not it is present in the input word lists or a British MySpell dictionary), include the flag `--disable_spell_check`.

Note, make sure the `--corpus_bigrams` argument includes the path to the bigram file output by `ngrams.py` and that the `--pwl_path` argument includes the path to the unigram personal word list output by `ngrams.py`.

Use the `--help` flag to get more information about remaining flags and arguments.

### TSV Files in Parallel

To run tokenization in parallel over a TSV file, use `prep_tsv.py` to split up the tsv file into `n` smaller files. Then, you can use the `parallel` command to tokenize all files in parallel. To split the file, run the following command:

```
./prep_tsv.py --tsv_file=TSV_PATH --num_splits=n
```

This will create a directory called `TSV_PATH-dir` containing about `n` tsv files (may contain more if number of lines in tsv file is not evenly divided by `n`). These files can be tokenized in parallel by executing the following command:
```
ls -d TSV_PATH-dir/*{0..9}.tsv | parallel --progress -j 64 "./run_tokenize.py --tsv_corpus={}"
```

This will tokenize each file in `TSV_PATH-dir` and write tokenized files to the same directory (with the suffix `-tok.tsv`). Note that you can modify the parameters passed to `run_tokenize.py` as described previously.

To merge the tokenized into one tsv file, use `prep_tsv.py` again with the following arguments:

`./prep_tsv.py TSV_PATH-dir/*-tok.tsv --rm_dir`

This command takes in all the tokenized files as input, merges them into a file with the same name as the original `TSV_PATH` but with `-tok.tsv` as its suffix. Specify the `--rm_dir` argument to remove the temporary `TSV_PATH-dir` directory which is no longer needed.

All of these steps can be done by running `./parallel-tokenize TSV_PATH`.

### Bigrams (optional)

To convert unigram data to bigram data, run `run_tokenize.py` with the flag `--bigrams`. Output data will be in bigram representation. Note that if you intend to use the Mallet wrapper code in `../topic-modeling/lda-tools` on a directory representation of the corpus, there is no need to convert your data to bigrams. Simply specify the `--bigrams_only` flag to `run_model.py` and the wrapper code will do the conversion.
