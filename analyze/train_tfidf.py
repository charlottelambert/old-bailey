#!/usr/bin/env python3
from gensim import models, corpora
from gensim.utils import simple_preprocess
import argparse, os, sys, time
from tqdm import tqdm
from gensim.corpora.mmcorpus import MmCorpus
import numpy as np
# from nltk import word_tokenize
sys.path.append('../')
from utils import *

def gensim_tfidf(args, pre, documents):
    """
        Code to run TF-IDF model on documents (using gensim).

        input:
            args (argparse object): input arguments
            pre (str): path to save model, corpus, and dictionary to
            documents (list): list of text from input documents

        returns tfidf model, corpus, and dictionary
    """
    # Create the Dictionary and Corpus
    print(timestamp() + " Building dictionary...", file=sys.stderr)
    mydict = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    mydict.filter_extremes(no_above=0.9, no_below=0)
    print(timestamp() + " Building corpus...", file=sys.stderr)
    corpus = [mydict.doc2bow(simple_preprocess(doc)) for doc in documents]

    print(timestamp() + " Creating tf-idf model... ", file=sys.stderr)
    # Create the TF-IDF model
    tfidf = models.TfidfModel(corpus, smartirs='ntc')

    tfidf.save(os.path.join(pre, "model"))
    mydict.save(os.path.join(pre, "dictionary"))
    MmCorpus.serialize(os.path.join(pre, "corpus"), corpus)
    print(timestamp() + " Model, corpus, and dictionary saved to directory " + pre, file=sys.stderr)

    return[tfidf, corpus, mydict]

def before_train(args):
    """
        Function to compile text from documents

        input:
            args (argparse object): input arguments

        returns prefix to save models to and list of all text from input files
    """
    print(timestamp() +  " Beginning tf-idf...", file=sys.stderr)
    pre = os.path.join(args.save_model_dir, "tf-idf", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    pre = pre.rstrip("/") + "/"
    print(pre)
    if not os.path.exists(pre):
        os.makedirs(pre)

    print(timestamp() + " Collecting files...", file=sys.stderr)

    files_dict, _ = order_files(args)
    documents = []
    for first_year, files in files_dict.items():
        joined_docs = []
        if args.tsv_corpus:
            joined_docs = [line.split("\t")[2] for line in files]
        else:
            # Merge all files from a year chunk into one file
            for file in files:
                with open(file) as f:
                    joined_docs.append(f.read())

        documents.append("\n".join(joined_docs))
    return [pre, documents]

def main(args):
    pre, documents = before_train(args)
    tfidf, corpus, mydict = gensim_tfidf(args, pre, documents)

    print(timestamp() + " Done!", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_corpus', type=str, default="", help='path to tsv file containing corpus')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-stats", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to include in each chunk of corpus (run tf-idf over each chunk)')
    args = parser.parse_args()
    main(args)
