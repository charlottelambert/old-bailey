#!/usr/bin/env python3
import nltk, json, os, sys, operator, argparse
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from utils import *

def valid_file(filename):
    """
        Determine if a file is within the valid range to construct bigrams,
        meaning between 1674 and October 1834 during which all proceedings and
        ordinary's accounts were manually typed.
    """
    year = get_year(filename, include_month=True)
    if year[0] < 1834: return True
    if year[0] == 1834 and year[1] < 10: return True
    return False

def make_bigram_dict(bigram_dict, text):
    words = [word.replace("\\", "") for sent in sent_tokenize(text) for word in word_tokenize(sent)]

    words = filter(lambda w: w not in ',-;?.():!', words)

    # Create your bigrams
    bgs = nltk.bigrams(words)

    # compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)
    bigram_dict.update({" ".join(k): v for (k,v) in fdist.items()})
    #for k,v in sorted(fdist.items(), key=operator.itemgetter(1)):
    #    print(k,v)

    return bigram_dict

def main(args):
    if args.output_path:
        out = args.output_path
    else: out = os.path.join(args.corpus_dir, "corpus_bigrams.json")

    if os.path.isfile(out) and not args.overwrite:
        print("Bigram file already exists. Include overwrite flag to recompute bigrams.", file=sys.stderr)
        exit(1)

    if args.file:
        text = open(args.corpus_dir).read()
        bigram_dict = make_bigram_dict({}, text)
    else:
        files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
                 if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]
        # Want only 1674 through Oct 1834
        if not args.disable_filter:
            print(timestamp(),"Filtering input files to all files between 1674 and October 1834...", file=sys.stderr)
            files = [file for file in files if valid_file(file)]

        print(timestamp(),"Computing bigrams...", file=sys.stderr)
        bigram_dict = {}
        for i in tqdm(range(len(files))):
            file = files[i]
            text = open(file).read()
            bigram_dict = make_bigram_dict(bigram_dict, text)

    bigram_dict = dict(sorted(bigram_dict.items(), key=operator.itemgetter(1)))

    # Write bigram dictionary to output file
    j = json.dumps(bigram_dict)
    f = open(out, "w")
    f.write(j)
    f.close()
    print(timestamp() + " Wrote bigram dictionary to", out, file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt", help='directory containing corpus')
    parser.add_argument('--output_path', type=str, default="", help='location to save output bigram dictionary')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--file', default=False, action="store_true", help='whether corpus_dir path is a file or not')
    parser.add_argument('--disable_filter', default=False, action="store_true", help='whether or not to disable filtering between 1674 and 1834')
    args = parser.parse_args()
    main(args)
