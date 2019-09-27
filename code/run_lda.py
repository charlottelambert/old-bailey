#!/usr/bin/env python3

###############################################################################
# run_lda.py
#
# Run LDA on input directory. Either use Mallet LDA or gensim's multicore based
# on input.
#
###############################################################################

import argparse
import time

import sys, os, click, gensim, datetime
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
# MALLET_PATH = os.environ.get("MALLET_PATH", "~/Mallet/bin/mallet")

def LDA_on_directory(args):
    if args.lda_type == "multicore":
        lda = gensim.models.ldamulticore.LdaMulticore
    elif args.lda_type == "mallet":
        lda = gensim.models.wrappers.LdaMallet

    print("Reading corpus.", file=sys.stderr)

    files = [f for f in os.listdir(args.corpus_dir)
             if os.path.isfile(os.path.join(args.corpus_dir, f))]

    # Compile list of lists of tokens
    texts = []
    for file in files:
        with open(os.path.join(args.corpus_dir, file)) as f:
            text = f.read().lower().replace("\n", " ").split(" ")

            # Changed: Also remove stop words from Mallet version
            stop_words = stop.modified_stop_words
            text = [word for word in text if word not in stop_words]
            texts.append(text)

    # If we want to include a mix of unigrams and bigrams or just bigrams
    if not args.unigrams_only:
        print("Finding bigrams.", file=sys.stderr)
        bigram = Phrases(texts, min_count=1) # Is this an appropriate value for min_count?
        bigram = Phraser(bigram)
        for idx in range(len(texts)):
            bigrams = bigram[texts[idx]]
            if args.bigrams_only:
                texts[idx] = [] # If we only want bigrams, remove all unigrams
            for token in bigrams:
                if '_' in token:
                    texts[idx].append(token)
                    if args.mixed_ngrams:
                        texts[idx].append(token) # Scale bigrams

    print("Building dictionary.", file=sys.stderr)

    dictionary = corpora.Dictionary(texts)
    if not args.mixed_ngrams:
        # Filter extremes if not doing only bigrams
        dictionary.filter_extremes(no_below=50, no_above=0.90)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # Prefix for running lda (modify if files should go to a different directory)
    pre = args.save_model_dir + "/" + time.strftime("%H:%M:%S") + args.lda_type + "."

    # Run specified model
    if lda == gensim.models.wrappers.LdaMallet:
        ldamodel = lda(args.mallet_path, corpus, num_topics=args.num_topics,
                       id2word=dictionary, optimize_interval=args.optimize_interval,
                       workers=12, iterations=args.num_iterations,
                       prefix=pre)
    elif lda == gensim.models.ldamulticore.LdaMulticore:
        ldamodel = lda(corpus, num_topics=args.num_topics,
                       id2word=dictionary, passes=200, alpha=20, workers=8,
                       prefix=pre)

    # Save model with timestamp
    ldamodel.save(pre + ".model")

    f = open(pre + "file_ordering.txt", "w+")
    text = ""
    for filename in files:
        text += filename + " "
    f.write(text[:-1])
    f.close()

    print("Done.", file=sys.stderr)

    return ldamodel.print_topics(num_topics=-1, num_words=20)
# _________________________________________________________________________

def main(corpus_dir, lda_type, unigrams_only, bigrams_only):
    print(LDA_on_directory(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mallet_path', type=str, default=os.environ.get("MALLET_PATH", "~/Mallet/bin/mallet"), help='path to where mallet is installed')
    parser.add_argument('--save_model_dir', type=str, default="../models/" + time.strftime("%Y%m%d"), help='path for saving model')
    parser.add_argument('--unigrams_only', default=False, action="store_true", help='whether or not to only include unigrams')
    parser.add_argument('--bigrams_only', default=False, action="store_true", help='whether or not to only include bigrams')
    parser.add_argument('--mixed_ngrams', default=False, action="store_true", help='whether or not to include both unigrams and bigrams')
    parser.add_argument('--corpus_dir', type=str, default="../data/sessionsPapers-txt", help='directory containing corpus')
    parser.add_argument('--lda_type', type=str, default="mallet", help='type of lda to run') # Include dynamic here?
    parser.add_argument('--num_topics', type=int, default=100, help='number of topics to find')
    parser.add_argument('--optimize_interval', type=int, default=10, help='number of topics to find')
    parser.add_argument('--num_iterations', type=int, default=1000, help='number of topics to find')
    args = parser.parse_args()
    main(args)
