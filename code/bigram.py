#!/usr/bin/env python3
import nltk, json, os, sys, operator
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


if os.path.isfile(sys.argv[1]):
    text = open(sys.argv[1]).read()
    bigram_dict = make_bigram_dict({}, text)
else:
    files = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1])
             if (os.path.isfile(os.path.join(sys.argv[1], f)) and f.endswith('.txt'))]
    # Want only 1674 through Oct 1834
    files = [file for file in files if valid_file(file)]

    bigram_dict = {}
    for i in tqdm(range(len(files))):
        file = files[i]
#    for file in files:
        text = open(file).read()
        bigram_dict = make_bigram_dict(bigram_dict, text)

bigram_dict = dict(sorted(bigram_dict.items(), key=operator.itemgetter(1)))

json = json.dumps(bigram_dict)
f = open("/work/clambert/thesis-data/corpus_bigrams.json", "w")
f.write(json)
f.close()
