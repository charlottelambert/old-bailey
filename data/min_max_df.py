#!/usr/bin/env python3
import argparse, sys, os
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("../")
from utils import *
from tqdm import tqdm
from joblib import Parallel, delayed

def parallel_process(items):
    file, documents, vocabulary, out_dir = items
    content = documents[file].split(" ")
    content = [word for word in content if word in vocabulary]
    with open(os.path.join(out_dir, file), 'w') as f:
        f.write(" ".join(content))

def main(args):
    print("Minimum document frequency:",args.min_df)
    print("Maximum document frequency:", args.max_df)
    files = [f for f in os.listdir(args.corpus_dir)
                     if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]
    print("Reading documents...", file=sys.stderr)
    documents = {}
    content = []
    for f in tqdm(files):
         with open(os.path.join(args.corpus_dir, f)) as file:
             text = file.read()
         documents[f] = text
         content.append(text)
    print("Building CountVectorizer...", file=sys.stderr)
    vectorizer = CountVectorizer(min_df=args.min_df, max_df=args.max_df)
    X = vectorizer.fit_transform(content)
    vocabulary = vectorizer.vocabulary_
    vocabulary = [v for v in vocabulary.keys()]
    print("Length of vocabulary:", len(vocabulary), file=sys.stderr)

    out_dir = args.corpus_dir.rstrip("/") + "-min_df_" + str(args.min_df) + "-max_df_" + str(args.max_df)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print("Writing updated files to", out_dir, file=sys.stderr)

    element_run = Parallel(n_jobs=-1)(delayed(parallel_process)((file, documents, vocabulary, out_dir)) for file in tqdm(files))
    print("Done!", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_data', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok.tsv", help='path to data in tsv format')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok", help='directory containing corpus')
    parser.add_argument('--min_df', default=0, type=int, help="minimum number of documents needed to contain a word")
    parser.add_argument('--max_df', default=1.0, type=float, help="maximum number of documents allowed to contain a word")
    args = parser.parse_args()
    main(args)
