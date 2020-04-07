#!/usr/bin/env python3
import os, argparse, csv, sys, copy, natsort, re, nltk, json
from tqdm import tqdm
from nltk.corpus import words
from flashtext import KeywordProcessor
import inflection as inf
from nltk.tag import pos_tag, pos_tag_sents
from gensim import models, corpora
from gensim.corpora.mmcorpus import MmCorpus
import numpy as np
from train_tfidf import *
from joblib import Parallel, delayed
import collections, functools, operator
from noun_counts import noun_counts
sys.path.append('../')
from utils import *
import matplotlib.pyplot as plt

english_words = KeywordProcessor()
latin_words = KeywordProcessor()

unk_words = set()

# Calculate number of named entities
def find_entities(all_tokens, stats_dict):
    # temp = [w.lower() for w in all_tokens]
    ret_dict = {k:v for k, v in stats_dict.items()}
    tagged_sent = pos_tag(all_tokens)
    tagged_nnp = [pair[0] for pair in tagged_sent if pair[1] == 'NNP']
    ret_dict["proper_nouns"] += len(tagged_nnp)
    # print(tagged_sent)
    # exit(0)
    found_entities = nltk.ne_chunk(tagged_sent, binary=True)
    # exit(0)
    named_entities = []
    for element in found_entities:
        try:
            if element.label() == "NE": named_entities.append(element)
        except AttributeError:
            continue
    ret_dict["num_entities"] += len(named_entities)
    return tagged_sent, ret_dict

def load_models(args):
    try:
        tfidf = models.TfidfModel.load(os.path.join(args.tfidf_model_dir_path, "model"))
        corpus = mm = MmCorpus(os.path.join(args.tfidf_model_dir_path, "corpus"))
        mydict = corpora.Dictionary.load(os.path.join(args.tfidf_model_dir_path, "dictionary"))
    except FileNotFoundError:
        print(timestamp(), "Tf-idf model directory path must contain model, corpus, and dictionary.", file=sys.stderr)
        exit(1)
    return tfidf, corpus, mydict

def get_top_words(args, doc_idx, tfidf=None, corpus=None, mydict=None):
    if not (tfidf and corpus and mydict):
        print(timestamp(), "get_top_words(): You must input a valid model, corpus, and dictionary.", file=sys.stderr)
        exit(1)

    weights = []
    doc = tfidf[corpus][doc_idx]
    weights = [[mydict[id], np.around(freq, decimals=2)] for id, freq in doc]
    top_words = natsort.natsorted(weights, key=lambda x: x[1], reverse=True)[:args.num_top_words]
    return [word[0] for word in top_words]

def update_tok_lists(all_tokens, list_to_check):
    out = []
    new_all = []
    for i in range(len(all_tokens)):
        if all_tokens[i] in list_to_check:
            out.append(all_tokens[i])
        else:
            new_all.append(all_tokens[i].lower()) # Lowercase because it isn't a proper noun anymore
    return [new_all, out]

# Get output of row
def get_stat_output(args, first_year, stats_dict, top_words, data_list):
    ret = [first_year]
    for count in data_list:
        # This tag isn't a percentage
        if count == "num_entities" or count == "total" or count == "proper_nouns":
            ret.append(stats_dict[count])
        elif count == "top_" + str(args.num_top_words) + "_words":
            ret.append(", ".join(top_words))
        # Divide percentages by total
        else: ret.append(round(stats_dict[count]/stats_dict["total"], 4))

    return ret

def stats_for_file(file, stats_dict):
    """
        For a given file, increment values in stats dict based on types of words
        in the file.

        return updated stats dictionary and flag indicating if valid. If not
        valid, will return original stats dict without updating.
    """
    backup = copy.deepcopy(stats_dict)

    # Define set if we're looking for only unique words
    if args.unique:
        words_seen = set()

    with open(file) as f:
        for line in f:
            line = line.replace("/", " ")

            # all_tokens: all words in line (excluding just punctuation)
            all_tokens = [tok for tok in line.split() if re.search('[a-zA-Z]', tok)]
            backup = [tok for tok in line.split() if re.search('[a-zA-Z]', tok)]

            # If only looking for unique words, don't add any that have already been processed
            if args.unique:
                all_tokens = list(set([tok for tok in all_tokens if tok not in words_seen]))
                words_seen.update(all_tokens)

            stats_dict['total'] += len(all_tokens)
            if args.count_entities:
                # propernouns: all proper nouns in the line
                tagged_sent, stats_dict = find_entities(all_tokens, stats_dict=stats_dict)
                # print(stats_dict)
                continue

            # Calculate capitalization statistics
            for word in all_tokens:
                if word.isupper():
                    stats_dict['upper'] += 1
                elif word.islower():
                    stats_dict['lower'] += 1
                else:
                    stats_dict['mixed'] += 1

            tagged_nnp = [pair[0] for pair in tagged_sent if pair[1] == 'NNP']

            all_tokens, propernouns = update_tok_lists(all_tokens, tagged_nnp)
            stats_dict['proper_nouns'] += len(propernouns)

            # english_words: all words in english in the line
            in_english = english_words.extract_keywords(" ".join(all_tokens))
            all_tokens, english_words_found = update_tok_lists(all_tokens, in_english)

            stats_dict['modern_english'] += len(english_words_found)

            # latin_words: all latin words in the line
            in_latin = latin_words.extract_keywords(" ".join(all_tokens))
            all_tokens, latin_words_found = update_tok_lists(all_tokens, in_latin)
            stats_dict['latin'] += len(latin_words_found)

            stats_dict['unk'] += len(all_tokens)
            unk_words.update([word for word in backup if word.lower() in all_tokens])
            try:
                assert stats_dict["modern_english"] + stats_dict["latin"] + stats_dict["old_english"] + stats_dict["proper_nouns"] + stats_dict["unk"] == stats_dict["total"]
                assert stats_dict["lower"] + stats_dict["upper"] + stats_dict["mixed"] == stats_dict["total"]
            except AssertionError:
                print(timestamp(), "Error: failed to process file. Skipping", file, file=sys.stderr)
                return [backup, 0]

    return [stats_dict, 1]

def init_stats_dict(data):
    """
        Initialize an empty stats dictionary.
    """
    stats_dict = {}
    for count in data:
        stats_dict[count] = 0
    return stats_dict


def graph_word_freqs(args, fd):
    """
        Function to generate and save a plot of word frequencies in corpus.

        input:
            args: argument object
            fd: nltk.FreqDist object containing word frequencies from whole corpus

    """
    path = os.path.join(args.corpus_dir, "word_freqs.png")
    plt.ion()
    fd.plot(30, title="Word Frequencies", cumulative=False)
    plt.savefig(path)
    plt.ioff()
    plt.show()
    print(timestamp() + " Done! Saved word frequency plot to", path, file=sys.stderr)

def find_basic_stats(args, files_dict):
    """
        Function finding basic statistics for all files in corpus.
    """
    print(timestamp() + " Starting statistics calculation...", file=sys.stderr)
    # First, find out how many files there are per year chunk
    stats_path = os.path.join(args.corpus_dir, "basic_stats.tsv")
    with open(stats_path, "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        stat_dict = {"stat_name": [], "num_docs":[],
                     "num_tokens":[], "num_types":[], "most_common_word":[]}
        corpus_fd = nltk.FreqDist()
        for start_year, files in files_dict.items():
            print(timestamp() + " Start year:", start_year, file=sys.stderr)
            slice_fd = nltk.FreqDist()
            stat_dict["stat_name"].append(start_year)
            stat_dict["num_docs"].append(len(files))
            num_tokens = 0
            types = set()
            for i in tqdm(range(len(files))):
                file = files[i]
                with open(file, "r") as f:
                    # Increment token count
                    toks = f.read().lower().split()
                    num_tokens += len(toks)
                    types.update(toks)
                    # Update frequency distribution for time slice
                    slice_fd.update(toks)
            # Update frequency distribution for whole corpus
            corpus_fd.update(slice_fd)

            stat_dict["num_tokens"].append(num_tokens)
            stat_dict["num_types"].append(len(types))
            # most common will return: [(word, frequency)]
            stat_dict["most_common_word"].append(slice_fd.most_common(1)[0][0])

        # Append stat values for entire corpus
        stat_dict["stat_name"].append("total")
        for row in stat_dict:
            if row == "most_common_word":
                stat_dict[row].append(corpus_fd.most_common(1)[0][0])
            elif not row == "stat_name":
                stat_dict[row].append(sum(stat_dict[row]))
            tsv_writer.writerow([row] + stat_dict[row])

    # Plot the word frequencies
    graph_word_freqs(args, corpus_fd)
    
    print(timestamp() + " Done! Wrote basic statistics to", stats_path, file=sys.stderr)


    # Then, find how many words (tokens and types) there are for each year chunk

def entity_helper(file):
    num_words = 0
    num_entities = 0
    with open(file, "r") as f:
        lines = [line.split() for line in f]
        for line in lines:
            tagged_tokens, cur_entities = find_entities(line)
            num_words += len(line)
            num_entities += cur_entities
    return num_words, num_entities

def main(args):

    data_list = ["modern_english", "old_english", "latin", "proper_nouns", "upper",
                 "lower", "mixed", "unk", "total", "num_entities"]

    # Calculate number of named entities in BNC
    if args.bnc_entities:
        # compile files into list
        print(timestamp(), "Finding entities for bnc files in directory", args.bnc_dir)
        files = [os.path.join(args.bnc_dir, f) for f in os.listdir(args.bnc_dir)
                 if (os.path.isfile(os.path.join(args.bnc_dir, f)))] # and f.endswith('.txt'))]


        element_run = Parallel(n_jobs=-1)(delayed(noun_counts)(files[i]) for i in tqdm(range(len(files))))

        for ret in element_run:
            num_proper_nouns = sum([ret[0] for ret in element_run])
            num_other_nouns = sum([ret[1] for ret in element_run])
            num_total = sum([ret[2] for ret in element_run])
        print(timestamp(), "Done! Number of proper nouns in BNC:", num_proper_nouns)
        print("Number of other nouns:", num_other_nouns)
        print("Number of total words:", num_total)
        print("Percent of proper nouns out of all nouns:", (num_proper_nouns/(num_proper_nouns + num_other_nouns)) * 100, "%")
        print("Percent of proper nouns in all text:", (num_proper_nouns/num_total) * 100, "%")
        exit(0)

    # Order files by year
    files_dict, _ = order_files(args)

    if args.basic_stats:
        find_basic_stats(args, files_dict)
        exit(0)

    # If we have a model to load, add fields to data_list and load model
    data_list.append("top_" + str(args.num_top_words) + "_words")

    if args.tfidf_model_dir_path:
        # load all models with path "model-XXXX" and put in dictionary
        tfidf, corpus, mydict = load_models(args)
        print(timestamp() + " Successfully loaded model, corpus, and dictionary from directory", args.tfidf_model_dir_path, file=sys.stderr)
    elif not args.disable_tfidf:
        # If not passed in a saved tfidf mode, run it now and save the output
        pre, documents = before_train(args)
        tfidf, corpus, mydict = gensim_tfidf(args, pre, documents)
    if args.count_entities: data_list = ["proper_nouns", "other_nouns", "num_entities", "total"]
    path_suff = ""
    if args.unique:
        path_suff += "-unique"
    # Add latin words to keyword processor
    with open(args.latin_dict) as f:
        latin_dict = f.read().split()
    latin_words.add_keywords_from_list(latin_dict)

    # Define an English dictionary depending on inputs
    if args.english_words:
        with open(args.english_words) as f:
            ewords = f.read().split("\n")
        english_words.add_keywords_from_list(ewords)
    else:
        english_words.add_keywords_from_list(words.words())

    # Initialize stats dict
    stats_dict = init_stats_dict(data_list)

    # Make output path
    base_stats_dir = os.path.join(args.corpus_dir, "../stats_dir/")
    if not os.path.exists(base_stats_dir):
        os.mkdir(base_stats_dir)

    stats_path = os.path.join(base_stats_dir, args.corpus_dir.rstrip('/') + path_suff + "-stats.tsv")
    with open(stats_path, "w") as f: # FIX OUTPUT DIRECTORY AND PATH
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(["start_year"] +  data_list)
        valid = 1
        doc_idx = 0
        for first_year, files in files_dict.items():
            # element_run = Parallel(n_jobs=-1)(delayed(stats_for_file)(files[i], init_stats_dict(data_list)) for i in tqdm(range(len(files))))

            for i in tqdm(range(len(files[:1]))):
                file_path = files[i]
                stats_dict, valid = stats_for_file(file_path, stats_dict)
            # valid = element_run[-1][1]
            # stat_list = [el[0] for el in element_run]
            # sum the values with same keys
            # stats_dict = dict(functools.reduce(operator.add, map(collections.Counter, stat_list)))
            if valid:
                if args.disable_tfidf:
                    top_words = []
                else:
                    top_words = get_top_words(args, doc_idx, tfidf, corpus, mydict)
                tsv_writer.writerow(get_stat_output(args, first_year, stats_dict, top_words, data_list))
                # else:
                #     tsv_writer.writerow([first_year] + [round(stats_dict[count]/stats_dict["total"], 4) for count in data_list])

            doc_idx += 1
    print(timestamp() + " Wrote statistics to", stats_path, file=sys.stderr)
    if args.print_unk:
        sorted = list(unk_words)
        sorted.sort()
        print("\n".join(sorted))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basic_stats', default=False, action='store_true', help='whether to find basic corpus stats only.')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-stats", help='directory containing corpus')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to calculate stats for')
    parser.add_argument('--num_top_words', type=int, default=10, help='number of top words to record')
    parser.add_argument('--latin_dict', type=str, default="/work/clambert/thesis-data/latin_dict.txt", help='text file containing latin dictionary')
    parser.add_argument('--english_words', type=str, default = "/work/clambert/thesis-data/bnc_lexicon.txt", help='optional path to file containing english words')
    parser.add_argument('--unique', default=False, action='store_true', help='whether or not to count only unique words')
    parser.add_argument('--tfidf_model_dir_path', type=str, default = "", help='path to tfidf model directory containing model to load.')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--count_entities', default=False, action='store_true', help='whether or not to count named entities in corpus and bnc')
    parser.add_argument('--print_unk', default=False, action='store_true', help='whether or not to print out unknown words')
    parser.add_argument('--bnc_dir', type=str, default="/work/clambert/thesis-data/parsed-bnc", help='path for calculating bnc entities')
    parser.add_argument('--bnc_entities', default=False, action='store_true', help='whether or not to calculate BNC entities')
    parser.add_argument('--disable_tfidf',  default=False, action='store_true')
    args = parser.parse_args()
    main(args)
