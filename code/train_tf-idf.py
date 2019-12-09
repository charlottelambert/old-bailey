#!/usr/bin/env python3
from gensim import models, corpora
from gensim.utils import simple_preprocess
import argparse, os, sys, time, natsort
from tqdm import tqdm
from gensim.corpora.mmcorpus import MmCorpus
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
# from nltk import word_tokenize


def build_vocab(documents):
    # build the vocabulary in one pass
    vocabulary = set()
    for doc in documents:
        words = doc.split()# tokenize(doc)
        vocabulary.update(words)

    vocabulary = list(vocabulary)
    # word_index = {w: idx for idx, w in enumerate(vocabulary)}

    VOCABULARY_SIZE = len(vocabulary)
    DOCUMENTS_COUNT = len(documents)
    return vocabulary
    # print(VOCABULARY_SIZE, DOCUMENTS_COUNT)


def sklearn_tfidf(args, pre, documents):
    stop_words = stopwords.words('english') + list(punctuation)
    print(timestamp, "Building vocabulary...", file=sys.stderr)
    vocabulary = build_vocab(documents)
    tfidf = TfidfVectorizer(stop_words=stop_words, vocabulary=vocabulary) #tokenizer=tokenize,

    print(timestamp, "Fitting the model...", file=sys.stderr)
    # Fit the TfIdf model
    # tfidf.fit([doc for doc in documents])
    # or:
    tfidf.fit_transform(documents) # ?

    # MAKE THIS SPLIT UP BY DOC IN DOCUMENTS! THIS IS JUST TF IDF OVER ALL

    print(timestamp, "Calculating top words...", file=sys.stderr)
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_n = 10
    top_features = [features[i] for i in indices[:top_n]]
    print(top_features)

    # # Transform a document into TfIdf coordinates
    # X = tfidf.transform([reuters.raw('test/14829')])
    #
    # # Check out some frequencies
    # print X[0, tfidf.vocabulary_['year']]                   # 0.0562524229373
    # print X[0, tfidf.vocabulary_['following']]              # 0.057140265658
    # print X[0, tfidf.vocabulary_['provided']]               # 0.0689364372666
    # print X[0, tfidf.vocabulary_['structural']]             # 0.0900802810906
    # print X[0, tfidf.vocabulary_['japanese']]               # 0.114492409303
    # print X[0, tfidf.vocabulary_['downtrend']]              # 0.111137191743


def gensim_tfidf(args, pre, documents):


    # Create the Dictionary and Corpus
    print(timestamp, "Building dictionary...", file=sys.stderr)
    mydict = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
    print(timestamp, "Building corpus...", file=sys.stderr)
    corpus = [mydict.doc2bow(simple_preprocess(doc)) for doc in documents]

    print(timestamp, "Creating tf-idf model... ", file=sys.stderr)
    # Create the TF-IDF model
    tfidf = models.TfidfModel(corpus, smartirs='ntc')

    # CHANGE NAMING CONVENTION!!!!
    tfidf.save(os.path.join(pre, "model"))
    mydict.save(os.path.join(pre, "dictionary"))
    MmCorpus.serialize(os.path.join(pre, "corpus"), corpus)
    print(timestamp, "Model, corpus, and dictionary saved to directory " + pre, file=sys.stderr)


def main(args):
    print(timestamp, "Beginning at " + time.strftime("%d/%m/%Y %H:%M "), file=sys.stderr)
    pre = os.path.join(args.save_model_dir, "tf-idf", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    pre = pre.rstrip("/") + "/"
    print(pre)
    if not os.path.exists(pre):
        os.makedirs(pre)

    print(timestamp, "Collecting files...", file=sys.stderr)

    files_dict = order_files(args)
    documents = []
    for first_year, files in files_dict.items():
        joined_docs = []
        # Merge all files from a year chunk into one file
        for file in files:
            with open(file) as f:
                joined_docs.append(f.read())

        documents.append("\n".join(joined_docs))

    sklearn_tfidf(args, pre, documents)
    # gensim_tfidf(args, pre, documents)

    print(timestamp, "Done!", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-stats", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to include in each chunk of corpus (run tf-idf over each chunk)')
    args = parser.parse_args()
    main(args)
