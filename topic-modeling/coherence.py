#!/usr/bin/env python3
import sys, os, argparse
from vis_topic_mallet import read_weighted_keys
from gensim.models import Word2Vec
from techknacq.corpus import Corpus
sys.path.append('..')
from utils import *
import math
from tqdm import tqdm

# https://www.overleaf.com/project/5e66a0594ea6e50001c76ffd
def vector_coherence(model, T):
    sum = 0
    acc = 0
    # Go through all distinct pairs in the topic
    for i, item_i in enumerate(T):
        w_i, weight_i = item_i
        for j, item_j in enumerate(T):
            w_j, weight_j = item_j
            if i == j: continue
            # Update sum
            sum += model.similarity(w_i, w_j)
            acc += 1

    return sum / acc


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

def load_docs(args):
    """
        Load documents and generate a corpus in particular format
    """
    files_dict, time_slices = order_files(args)
    corpus = Corpus(args.corpus_dir)
    new_corpus = {}
    for file, doc in corpus.docs.items():
        new_corpus[file] = set(doc.text().lower().split())
    return new_corpus


def main(args):
    # Extract topics from each model
    print("Getting topics...", file=sys.stderr, end=' ')
    # Open weighted key files
    weighted_keys = open(args.weighted_keys, 'r')
    # Extract topics and weights
    topics = read_weighted_keys(weighted_keys)
    print("Done!", file=sys.stderr)
    if args.method == 'umass':
        print("Generating corpus...", file=sys.stderr, end=' ')
        # Generate corpus from documents
        # export PYTHONPATH=/home/clambert/thesis/topic-modeling/lda-tools/lib:$PYTHONPATH
        corpus = load_docs(args)
        print("Done!", file=sys.stderr)

        # print(D(corpus, "murther", second_word="murder"))
        # NOW IMPLEMENT THAT MASS THING!!
        # exit(0)
    elif args.method == 'vectors':
        print("Loading word2vec model from path", args.word2vec_model, file=sys.stderr, end=' ')
        model = Word2Vec.load(args.word2vec_model)
        print("Done!", file=sys.stderr)
        # print(model.wv.vocab)
        # print(len(model.wv.vocab))
        # exit(0)

    print("Calculating coherence for", len(topics) ,"topics...", file=sys.stderr)
    c_value = 0
    for id, topic in tqdm(topics.items()):
        if args.method == 'umass':
            c_value += umass_coherence(topic, corpus)
        elif args.method =='vectors':
            c_value += vector_coherence(model, topic)
    print("Average topic coherence:", c_value / len(topics))
    print("Done!", file=sys.stderr)

# try: ./coherence /work/clambert/models/lda/20200317/10-42-35/weighted-keys.txt --method=vectors
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weighted_keys', type=str, help='path to weighted_keys.txt file to load')
    parser.add_argument('--corpus_dir', type=str, default='/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-lower', help='path to corpus path')
    parser.add_argument('--method', type=str, default='umass', help='method to use when calculating coherence')
    parser.add_argument('--word2vec_model', type=str, default='/work/clambert/models/word2vec/2020-04-01/09-45-23/1674.model', help='path to word2vec model to use in vector method')
    args = parser.parse_args()
    main(args)
