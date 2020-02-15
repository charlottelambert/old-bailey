#!/usr/bin/env python3
import os, argparse, csv, sys, copy, natsort, re
from tqdm import tqdm
from nltk.corpus import words
from flashtext import KeywordProcessor
import inflection as inf
from nltk.tag import pos_tag
from utils import *
from gensim import models, corpora
from gensim.corpora.mmcorpus import MmCorpus
import numpy as np
from train_tfidf import *

english_words = KeywordProcessor()
latin_words = KeywordProcessor()

data_list = ["modern_english", "old_english", "latin", "proper_nouns", "upper", "lower", "mixed", "unk", "total"]

unk_words = set()

def load_models_old(args):
    model_dict = {}
    files = [os.path.join(args.tfidf_model_dir_path, f) for f in os.listdir(args.tfidf_model_dir_path)
             if (os.path.isfile(os.path.join(args.tfidf_model_dir_path, f)))]
    all_models = [path for path in files if "model" in os.path.basename(path)]
    # print("ALL",all_models)
    for model in all_models:
        # print(model)
        if model == "model":
            print("Please re-run train_tfidf.py and provide new path.", file=sys.stderr)
            exit(1)
        try:
            year = int(os.path.basename(model).split("-")[1])
        except:
            print("Error with naming of file " + model + ". Files should be in format model-YYYY,", file=sys.stderr)
            exit(1)
        # cur_year_files = [path for path in files if path.endswith(str(year))]
        try:
            corpus = mm = MmCorpus(os.path.join(args.tfidf_model_dir_path, "corpus-" + str(year)))
            tfidf = models.TfidfModel.load(os.path.join(args.tfidf_model_dir_path, "model-" + str(year)))
            mydict = corpora.Dictionary.load(os.path.join(args.tfidf_model_dir_path, "dictionary-" + str(year)))
        except FileNotFoundError:
            print("Tf-idf model directory path must contain model, corpus, and dictionary with year as suffix.", file=sys.stderr)
            exit(1)

        model_dict[year] = {"corpus":corpus, "model":tfidf, "dictionary":mydict}
    return model_dict

def load_models(args):
    try:
        tfidf = models.TfidfModel.load(os.path.join(args.tfidf_model_dir_path, "model"))
        corpus = mm = MmCorpus(os.path.join(args.tfidf_model_dir_path, "corpus"))
        mydict = corpora.Dictionary.load(os.path.join(args.tfidf_model_dir_path, "dictionary"))
    except FileNotFoundError:
        print("Tf-idf model directory path must contain model, corpus, and dictionary.", file=sys.stderr)
        exit(1)
    return tfidf, corpus, mydict

def get_top_words(args, doc_idx, tfidf=None, corpus=None, mydict=None):
    if not (tfidf and corpus and mydict):
        print("get_top_words(): You must input a valid model, corpus, and dictionary.", file=sys.stderr)
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

            # If only looking for unique words, don't add any that have already been processed
            if args.unique:
                all_tokens = list(set([tok for tok in all_tokens if tok not in words_seen]))
                words_seen.update(all_tokens)

            stats_dict['total'] += len(all_tokens)

            # Calculate capitalization statistics
            for word in all_tokens:
                if word.isupper():
                    stats_dict['upper'] += 1
                elif word.islower():
                    stats_dict['lower'] += 1
                else:
                    stats_dict['mixed'] += 1

            # propernouns: all proper nouns in the line
            tagged_sent = pos_tag(all_tokens)
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
            unk_words.update(all_tokens)
            try:
                assert stats_dict["modern_english"] + stats_dict["latin"] + stats_dict["old_english"] + stats_dict["proper_nouns"] + stats_dict["unk"] == stats_dict["total"]
                assert stats_dict["lower"] + stats_dict["upper"] + stats_dict["mixed"] == stats_dict["total"]
            except AssertionError:
                print("Error: failed to process file. Skipping", file, file=sys.stderr)
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


def find_basic_stats(args, files_dict):
    print(timestamp() + " Starting statistics calculation...", file=sys.stderr)
    # First, find out how many files there are per year chunk
    stats_path = os.path.join(args.corpus_dir, "basic_stats.tsv")
    with open(stats_path, "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        stat_dict = {"stat_name": [], "num_docs":[],
                     "num_tokens":[], "num_types":[]}

        for start_year, files in files_dict.items():
            print(timestamp() + " Start year:", start_year, file=sys.stderr)

            stat_dict["stat_name"].append(start_year)
            stat_dict["num_docs"].append(str(len(files)))
            num_tokens = 0
            types = set()
            for i in tqdm(range(len(files))):
                file = files[i]
                with open(file, "r") as f:
                    toks = f.read().split()
                    num_tokens += len(toks)
                    types.update(toks)

            stat_dict["num_tokens"].append(str(num_tokens))
            stat_dict["num_types"].append(str(len(types)))

            # item 1674 1774 1874
            # num_documents x y z
            # num_tokens
            # num_types
        for row in stat_dict:
            stat_dict["stat_name"].append("total")
            stat_dict[row].append(sum(stat_dict[row]))
            tsv_writer.writerow([row] + stat_dict[row])
    print(timestamp() + " Done! Wrote basic statistics to", stats_path, file=sys.stderr)


    # Then, find how many words (tokens and types) there are for each year chunk

def main(args):
    # Order files by year
    files_dict = order_files(args)

    if args.basic_stats:
        find_basic_stats(args, files_dict)
        exit(0)

    # If we have a model to load, add fields to data_list and load model
    data_list.append("top_" + str(args.num_top_words) + "_words")

    if args.tfidf_model_dir_path:
        # load all models with path "model-XXXX" and put in dictionary
        tfidf, corpus, mydict = load_models(args)
        print(timestamp() + " Successfully loaded model, corpus, and dictionary from directory", args.tfidf_model_dir_path, file=sys.stderr)
    else:
        # If not passed in a saved tfidf mode, run it now and save the output
        pre, documents = before_train(args)
        tfidf, corpus, mydict = gensim_tfidf(args, pre, documents)

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
            for i in tqdm(range(len(files))):
                file_path = files[i]
                stats_dict, valid = stats_for_file(file_path, stats_dict)

            if valid:
                top_words = get_top_words(args, doc_idx, tfidf, corpus, mydict)
                tsv_writer.writerow([first_year] + [round(stats_dict[count]/stats_dict["total"], 4) for count in data_list] + [", ".join(top_words)])
                # else:
                #     tsv_writer.writerow([first_year] + [round(stats_dict[count]/stats_dict["total"], 4) for count in data_list])

            doc_idx += 1
    print(timestamp() + " Wrote statistics to", stats_path, file=sys.stderr)
    print("\n".join(list(unk_words)))

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
    args = parser.parse_args()
    main(args)
