#!/usr/bin/env python3
from gensim import models, corpora
from gensim.utils import simple_preprocess
import argparse, os, sys, time, natsort
from tqdm import tqdm
from gensim.corpora.mmcorpus import MmCorpus
from utils import *


def main(args):
    print("Beginning at " + time.strftime("%d/%m/%Y %H:%M "), file=sys.stderr)
    pre = os.path.join(args.save_model_dir, "tf-idf", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    pre = pre.rstrip("/") + "/"
    print(pre)
    if not os.path.exists(pre):
        os.makedirs(pre)
    print("Building corpus...", file=sys.stderr)

    files_dict = order_files(args)
    for first_year, files in files_dict.items():
        documents = []
        for file in files:
            with open(file) as f:
                documents.append(f.read())
        # documents = " ".join(documents) # Makes one corpus for entire year split
        # Create the Dictionary and Corpus
        mydict = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
        # mydict = corpora.Dictionary([simple_preprocess(documents)])
        corpus = [mydict.doc2bow(simple_preprocess(doc)) for doc in documents]
        # corpus = [mydict.doc2bow(simple_preprocess(documents)) ]

        print("Creating tf-idf model... ", file=sys.stderr)
        # Create the TF-IDF model
        tfidf = models.TfidfModel(corpus, smartirs='ntc')

        # CHANGE NAMING CONVENTION!!!!
        tfidf.save(os.path.join(pre, "model-" + str(first_year)))
        mydict.save(os.path.join(pre, "dictionary-" + str(first_year)))
        MmCorpus.serialize(os.path.join(pre, "corpus-" + str(first_year)), corpus)
    print("Model, corpus, and dictionary saved to directory " + pre, file=sys.stderr)


    # # Show the Word Weights in Corpus
    # for doc in corpus:
    #     print([[mydict[id], freq] for id, freq in doc])
    #
    #


    print("Done! Ending at " + time.strftime("%d/%m/%Y %H:%M ") , file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-dh", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--year_split', type=int, default=100,help='number of years to include in each chunk of corpus (run tf-idf over each chunk)')
    args = parser.parse_args()
    main(args)
