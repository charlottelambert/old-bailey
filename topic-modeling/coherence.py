#!/usr/bin/env python3
import sys, os, argparse
from vis_topic_mallet import read_weighted_keys
from gensim.models import Word2Vec
from techknacq.corpus import Corpus
sys.path.append('..')
from utils import *
import math
from tqdm import tqdm

# http://qpleple.com/topic-coherence-to-evaluate-topic-models/
# UMASS!
def coherence(T, corpus):
    """
        Find the coherence of a topic
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
            val = (counts["both"] + 1) / counts[w_i]
            # Update sum
            sum += math.log(val, 10)
    return sum

def load_docs(args):
    files_dict, time_slices = order_files(args)
    corpus = Corpus(args.corpus_dir)
    return corpus

def D(corpus, word, second_word=None):
    counts = {word:0}
    if second_word:
        counts["both"] = 0
    for doc_name, doc in corpus.docs.items():
        # Update D(w_i)
        if word in doc.text():
            counts[word] += 1
            # Update D(w_i, w_j)
            if second_word and second_word in doc.text():
                counts["both"] += 1
    return counts

def main(args):
    # Extract topics from each model
    # export PYTHONPATH=/home/clambert/thesis/topic-modeling/lda-tools/lib:$PYTHONPATH
    corpus = load_docs(args)
    # print(D(corpus, "murther", second_word="murder"))
    # NOW IMPLEMENT THAT MASS THING!!
    # exit(0)
    print("Getting topics...", file=sys.stderr, end=' ')
    # Open weighted key files
    weighted_keys = open(args.weighted_keys, 'r')
    # Extract topics and weights
    topics = read_weighted_keys(weighted_keys)
    print("Done!", file=sys.stderr)
    print("Calculating coherence for", len(topics) ,"topics...", file=sys.stderr)
    for id, topic in tqdm(topics.items()):
        print(coherence(topic, corpus))
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weighted_keys', type=str, help='path to weighted_keys.txt file to load')
    parser.add_argument('--corpus_dir', type=str, default='/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok', help='path to corpus path')
    parser.add_argument('--model_path', type=str, help='path to Word2Vec model to load', default='/work/clambert/models/word2vec/2020-03-22/16-23-47/1674.model')
    args = parser.parse_args()
    main(args)
