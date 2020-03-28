#!/usr/bin/env python3
import nltk, json, os, sys, operator, argparse, copy
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
sys.path.append('../')
from utils import *

stop_words = set(stopwords.words('english'))

def valid_file(filename):
    """
        Determine if a file is within the valid range to construct bigrams,
        meaning between 1674 and October 1834 during which all proceedings and
        ordinary's accounts were manually typed.
    """
    # All files from London Lives should be true
    if len(filename) > 14: return True
    year = get_year(filename, include_month=True)
    if year[0] < 1834: return True
    if year[0] == 1834 and year[1] < 10: return True
    return False

def make_ngram_dicts(unigram_dict, bigram_dict, text):
    words = [word.replace("\\", "") for sent in sent_tokenize(text) for word in word_tokenize(sent) if word not in stop_words]

    words = filter(lambda w: w not in ',-;?.():!', words)
    uni_words = list(copy.deepcopy(words))

    # Create your bigrams
    bgs = nltk.bigrams(words)
    # compute frequency distribution for all the bigrams in the text
    bigram_dist = nltk.FreqDist(bgs)
    bigram_dict.update({" ".join(k): v for (k,v) in bigram_dist.items()})

    # compute frequency distribution for all the unigrams in the text
    unigram_dist = nltk.FreqDist(uni_words)
    unigram_dict.update({k: v for (k,v) in unigram_dist.items()})
    return (unigram_dict, bigram_dict)

def main(args):
    prefix = os.path.dirname(args.corpus_dir) if args.file else args.corpus_dir
    uni_out = os.path.join(prefix, "corpus_unigrams.json")
    bi_out = os.path.join(prefix, "corpus_bigrams.json")
    print(bi_out)
    if os.path.isfile(bi_out) and not args.overwrite:
        print("Bigram file already exists. Include overwrite flag to recompute bigrams.", file=sys.stderr)
        exit(1)
    if os.path.isfile(uni_out) and not args.overwrite:
        print("Unigram file already exists. Include overwrite flag to recompute unigrams.", file=sys.stderr)
        exit(1)

    if args.file:
        text = open(args.corpus_dir).read()
        unigram_dict, bigram_dict = make_ngram_dicts({}, {}, text)
    else:
        files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
                 if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]
        # Want only 1674 through Oct 1834
        if not args.disable_filter:
            print(timestamp(),"Filtering input files to all files between 1674 and October 1834...", file=sys.stderr)
            files = [file for file in files if valid_file(file)]

        print(timestamp(),"Computing unigrams and bigrams...", file=sys.stderr)
        unigram_dict = {}
        bigram_dict = {}
        for i in tqdm(range(len(files))):
            file = files[i]
            text = open(file).read()
            unigram_dict, bigram_dict = make_ngram_dicts(unigram_dict, bigram_dict, text)

    unigram_dict = dict(sorted(unigram_dict.items(), key=operator.itemgetter(1)))
    bigram_dict = dict(sorted(bigram_dict.items(), key=operator.itemgetter(1)))

    # Write bigram dictionary to output file
    b = json.dumps(bigram_dict)
    f = open(bi_out, "w")
    f.write(b)
    f.close()
    print(timestamp() + " Wrote bigram dictionary to", bi_out, file=sys.stderr)
    # Write unigram dictionary to output file
    u = json.dumps(unigram_dict)
    f = open(uni_out, "w")
    f.write(u)
    f.close()
    print(timestamp() + " Wrote unigram dictionary to", uni_out, file=sys.stderr)
    pwl_out = os.path.join(prefix, "unigram_pwl.txt")
    with open(pwl_out, "w") as f:
        f.write("\n".join(word for word, freq in unigram_dict.items()))
    print(timestamp() + " Wrote personal word list of unigrams to", pwl_out, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt", help='directory containing corpus')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--file', default=False, action="store_true", help='whether corpus_dir path is a file or not')
    parser.add_argument('--disable_filter', default=False, action="store_true", help='whether or not to disable filtering between 1674 and 1834')
    args = parser.parse_args()
    main(args)
