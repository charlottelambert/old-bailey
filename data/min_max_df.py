#!/usr/bin/env python3
import argparse, sys, os
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("../")
from utils import *
from tqdm import tqdm
from joblib import Parallel, delayed

def parallel_process(items):
    file, documents, vocabulary, out_dir = items
    if os.path.exists(os.path.join(out_dir, file)): return
    content = documents[file].split(" ")
    content = [word for word in content if word in vocabulary]
    with open(os.path.join(out_dir, file), 'w') as f:
        f.write(" ".join(content))

def main(args):
#    print("Minimum document frequency:",args.min_df)
    if args.filepath: out_dir = os.path.dirname(args.filepath).rstrip("/") + "-min_df_" + str(args.min_df) + "-max_df_" + str(args.max_df)
    else: out_dir = args.corpus_dir.rstrip("/") + "-min_df_" + str(args.min_df) + "-max_df_" + str(args.max_df)
#    print("Maximum document frequency:", args.max_df)
    if args.filepath: files = [os.path.basename(args.filepath)]
    else:
        files = [f for f in os.listdir(args.corpus_dir)
                         if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]
#    print("Reading documents...", file=sys.stderr)
    base_dir = args.corpus_dir if not args.filepath else os.path.dirname(args.filepath)
    documents = {}
    content = []
    for f in files:
         with open(os.path.join(base_dir, f)) as file:
             text = file.read()
         documents[f] = text
         content.append(text)

    if args.vocab_path:
#         print("Loading vocabulary from", args.vocab_path, file=sys.stderr)
         with open(args.vocab_path) as f:
             vocabulary = f.read().split("\n")
    else:
        print("Building CountVectorizer...", file=sys.stderr)
        vectorizer = CountVectorizer(min_df=args.min_df, max_df=args.max_df)
        X = vectorizer.fit_transform(content)
        vocabulary = vectorizer.vocabulary_
        vocabulary = [v for v in vocabulary.keys()]
        with open(out_dir + "-vocab", "w") as f:
            f.write("\n".join(vocabulary))

        print("Writing vocabulary to", out_dir + "-vocab", file=sys.stderr)
#    print("Length of vocabulary:", len(vocabulary), file=sys.stderr)

    try: os.makedirs(out_dir)
    except: pass
#    print("Writing updated files to", out_dir, file=sys.stderr)
    if args.filepath:
        parallel_process((os.path.basename(args.filepath), documents, vocabulary, out_dir))
        exit(0)
    # Otherwise, do all files in corpus
    for file in tqdm(files):
        parallel_process((file, documents, vocabulary, out_dir))
#    element_run = Parallel(n_jobs=-1)(delayed(parallel_process)((file, documents, vocabulary, out_dir)) for file in tqdm(files))
#    print("Done!", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_data', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok.tsv", help='path to data in tsv format')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok", help='directory containing corpus')
    parser.add_argument('--filepath', type=str, default="", help='path to single file to process')
    parser.add_argument('--min_df', default=0, type=int, help="minimum number of documents needed to contain a word")
    parser.add_argument('--max_df', default=1.0, type=float, help="maximum number of documents allowed to contain a word")
    parser.add_argument('--vocab_path', default="", type=str, help='path to vocabulary file (one word per line)')
    args = parser.parse_args()
    main(args)
