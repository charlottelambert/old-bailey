#!/usr/bin/env python3

###############################################################################
# run_lda.py
#
# Run LDA on input directory. Either use Mallet LDA or gensim's multicore based
# on input.
#
# Specify -u flag to run LDA with only unigrams.
# Specify -b flag to run LDA with only bigrams.
# Default (no flag) runs LDA with a mix of unigrams and bigrams.
#
###############################################################################

import sys
import os
import click
import gensim
import datetime
from os import listdir
import custom_stop_words as stop
from gensim import corpora, models
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import get_tmpfile
from gensim.corpora import MalletCorpus
from gensim.models.phrases import Phrases, Phraser

# Where is Mallet installed? Can be overridden with MALLET_PATH environment
# variable, e.g.,
#     MALLET_PATH=~/opt/mallet/bin/mallet ./run_lda.py
MALLET_PATH = os.environ.get("MALLET_PATH", "~/Mallet/bin/mallet")


def LDA_on_directory(book_directory, lda, ngrams=0):
    print("Reading corpus.", file=sys.stderr)

    files = [f for f in os.listdir(book_directory)
             if os.path.isfile(os.path.join(book_directory, f))]

    # Compile list of lists of tokens
    texts = []
    for file in files:
        with open(os.path.join(book_directory, file)) as f:
            text = f.read().lower().replace("\n", " ").split(" ")

            # Changed: Also remove stop words from Mallet version
            stop_words = stop.modified_stop_words
            text = [word for word in text if word not in stop_words]
            texts.append(text)

    # If we want to include a mix of unigrams and bigrams or just bigrams
    if ngrams != 1:
        print("Finding bigrams.", file=sys.stderr)
        bigram = Phrases(texts, min_count=1) # Is this an appropriate value for min_count?
        bigram = Phraser(bigram)
        for idx in range(len(texts)):
            bigrams = bigram[texts[idx]]
            if ngrams == 2:
                texts[idx] = [] # If we only want bigrams, remove all unigrams
            for token in bigrams:
                if '_' in token:
                    texts[idx].append(token)
                    if ngrams == 0:
                        texts[idx].append(token) # Scale bigrams

    print("Building dictionary.", file=sys.stderr)

    dictionary = corpora.Dictionary(texts)
    if ngrams != 2:
        # Filter extremes if not doing only bigrams
        dictionary.filter_extremes(no_below=50, no_above=0.90)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # Prefix for running lda (modify if files should go to a different directory)
    pre = "/home/clambert/RA-Fall-2018/models/"
    # Timestamp for labeling files
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Run specified model
    if lda == gensim.models.wrappers.LdaMallet:
        pre += "mallet_" + now + "."
        ldamodel = lda(MALLET_PATH, corpus, num_topics=200,
                       id2word=dictionary, optimize_interval=10,
                       workers=12, iterations=1000,
                       prefix=pre)
    elif lda == gensim.models.ldamulticore.LdaMulticore:
        pre += "multicore_" + now + "."
        ldamodel = lda(corpus, num_topics=200,
                       id2word=dictionary, passes=200, alpha=20, workers=8,
                       prefix=pre)

    # Save model with timestamp
    ldamodel.save(pre + "model")

    f = open(pre + "file_ordering.txt", "w+")
    text = ""
    for filename in files:
        text += filename + " "
    f.write(text[:-1])
    f.close()

    print("Done.", file=sys.stderr)

    return ldamodel.print_topics(num_topics=-1, num_words=20)
# _________________________________________________________________________

@click.command()
@click.argument('book_directory', type=click.Path())
@click.option('-u', '--unigrams', 'unigrams_only', is_flag=True)
@click.option('-b', '--bigrams', 'bigrams_only', is_flag=True)
@click.argument('lda_type', type=click.Path())
def main(book_directory, lda_type, unigrams_only, bigrams_only):
    if lda_type == "multicore":
        lda = gensim.models.ldamulticore.LdaMulticore
    elif lda_type == "mallet":
        lda = gensim.models.wrappers.LdaMallet
    # Determine the types of ngrams to use
    ngrams = 0
    if unigrams_only:
        ngrams = 1
    elif bigrams_only:
        ngrams = 2
    print(LDA_on_directory(book_directory, lda, ngrams))

if __name__ == '__main__':
    main()
