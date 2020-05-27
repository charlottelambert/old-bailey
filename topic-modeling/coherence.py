#!/usr/bin/env python3
import sys, os, argparse
from vis_topic_mallet import read_weighted_keys
from gensim.models import Word2Vec
from techknacq.corpus import Corpus
sys.path.append('..')
from utils import *
import math
from tqdm import tqdm
from optparse import OptionParser

def vector_coherence(model, T):
    """
        Calculate topic coherence of a topic given word2vec model.

        input:
            model (w2v model): model containing word similarities
            T (list): list of words representing a topic

        return topic coherence for topic T
    """
    sum = 0
    acc = 0
    # Go through all distinct pairs in the topic
    for i, item_i in enumerate(T):
        w_i, weight_i = item_i
        for j, item_j in enumerate(T):
            w_j, weight_j = item_j
            if i == j: continue
            # Update sum
            try:
                sum += model.similarity(w_i, w_j)
                acc += 1
            except:
                print("One of these words is not in model:",w_i, w_j)
                continue
    try: ret = sum/acc
    except:
        ret = 0
        print("Problem with topic:",T". Skipping...")
    return ret

# http://qpleple.com/topic-coherence-to-evaluate-topic-models/
def umass_coherence(T, corpus):
    """
        Find the UMass coherence of a topic
        input:
            T (list of strings): topic to find coherence of
    """

    sum = 0
    # Go through all distinct pairs in the topic
    for i, item_i in enumerate(T):
        w_i, weight_i = item_i
        for j, item_j in enumerate(T):
            w_j, weight_j = item_j
            if not (i < j): continue
            # Calculate how many documents w_i appears in and how many
            # documents w_i and w_j both appear in
            counts = D(corpus, w_i, second_word=w_j)
            # Update sum
            val = (counts["both"] + 1) / counts[w_i]
            sum += math.log(val, 10)
    return sum

def D(corpus, word, second_word=None):
    """
        Function to count how many documents a word appears in. If provided a
        second word, will also calculate the number of documents both words
        appear in.
    """
    counts = {word:0}
    if second_word:
        counts["both"] = 0
    # Need corpus to be in format: {filename: file_contents.split()}
    for doc_name, doc in corpus.items():
        # Update D(w_i)
        text = doc
        if word in text:
            counts[word] += 1
            # Update D(w_i, w_j)
            if second_word and second_word in text:
                counts["both"] += 1
    return counts

def load_docs(options):
    """
        Load documents and generate a corpus in particular format
    """
    files_dict, time_slices = order_files(options)
    corpus = Corpus(options.corpus_dir)
    new_corpus = {}
    for file, doc in corpus.docs.items():
        new_corpus[file] = set(doc.text().lower().split())
    return new_corpus


def main(options, args):
    if options.method == 'vectors':
        print("Loading word2vec model from path", options.word2vec_model, file=sys.stderr, end=' ')
        model = Word2Vec.load(options.word2vec_model)
        print("Done!", file=sys.stderr)

    for wk in args:
        # Extract topics from each model
        print("Getting topics...", file=sys.stderr, end=' ')
        # Open weighted key files
        weighted_keys = open(wk, 'r')
        # Extract topics and weights
        topics = read_weighted_keys(weighted_keys)
        print("Done!", file=sys.stderr)
        if options.method == 'umass':
            print("Generating corpus...", file=sys.stderr, end=' ')
            # Generate corpus from documents
            corpus = load_docs(options)
            print("Done!", file=sys.stderr)

        print("Calculating coherence for", len(topics) ,"topics...", file=sys.stderr)
        c_value = 0
        for id, topic in tqdm(topics.items()):
            if options.method == 'umass':
                c_value += umass_coherence(topic, corpus)
            elif options.method =='vectors':
                c_value += vector_coherence(model, topic)
        print("Weighted keys file:", wk)
        print("Average topic coherence:", c_value / len(topics))
        print("Done!", file=sys.stderr)

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] weighted_keys1 weighted_keys2 ...")
    parser.add_option('--method', type=str, default='vectors', help='method to use when calculating coherence')
    parser.add_option('--word2vec_model', type=str, default='', help='path to word2vec model to use in vector method')
    (options, args) = parser.parse_args()
    if(len(args) < 1):
        parser.error( "Must specify at least one weighted keys file" )
    main(options, args)
