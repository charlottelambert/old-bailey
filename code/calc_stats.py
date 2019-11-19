#!/usr/bin/env python3
import os, argparse, csv, sys
from tqdm import tqdm
from nltk.corpus import words
from flashtext import KeywordProcessor
import inflection as inf
from nltk.tag import pos_tag

english_words = KeywordProcessor()
english_words.add_keywords_from_list(words.words())
latin_words = KeywordProcessor()

data_list = ["modern_english", "old_english", "latin", "unk", "total", "proper_nouns"]

def stats_for_file(file, stats_dict):
    with open(file) as f:
        for line in f:
            # Increment value for all tokens
            all_tokens = line.split() # SHOULD I JUST BE CHECKING FOR UNIQUE TOKENS?
            stats_dict['total'] += len(all_tokens)
            # print("\n", all_tokens)
            # Find proper nouns in this line
            tagged_sent = pos_tag(all_tokens)
            propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
            # print(propernouns)
            stats_dict['proper_nouns'] += len(propernouns)
            remaining_words = [tok.lower() for tok in all_tokens if tok not in propernouns]
            # THEN lower
            # then check rest

            # line = " ".join([inf.singularize(tok) for tok in all_tokens])

            english_words_found = english_words.extract_keywords(" ".join(remaining_words))
            # print(english_words_found)
            english_words_found = [word.lower() for word in english_words_found]
            stats_dict['modern_english'] += len(english_words_found)

            remaining_words = [word for word in remaining_words if word not in english_words_found]
            # print(remaining_words)
            latin_words_found = latin_words.extract_keywords(" ".join(remaining_words))
            # print(latin_words_found)
            stats_dict['latin'] += len(latin_words_found)
            remaining_words = [word for word in remaining_words if word not in latin_words_found]
            stats_dict['unk'] += len(remaining_words)
            # print(remaining_words)
            # print("\n")
            # exit(1)

    return stats_dict

def main(args):
    with open(args.latin_dict) as f:
        latin_dict = f.read().split()
    latin_words.add_keywords_from_list(latin_dict)

    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]

    # Initialize stats dict
    stats_dict = {}
    for count in data_list:
        stats_dict[count] = 0

    with open("stats.tsv", "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(data_list)

        for i in tqdm(range(len(files))):
            stats_dict = stats_for_file(files[i], stats_dict)
            try:
                # print(stats_dict)
                assert stats_dict["modern_english"] + stats_dict["latin"] + stats_dict["old_english"] + stats_dict["proper_nouns"] + stats_dict["unk"] == stats_dict["total"]
            except AssertionError:
                print("Incorrectly counted words for file " + files[i] + ". Aborting.", file=sys.stderr)

        # Write all data to tsv file (calculates over entire corpus)
        tsv_writer.writerow([stats_dict[count] for count in data_list])

    print("Wrote statistics to stats.tsv ")
    # Estimate what amount of text is proper nouns, Latin, historical English,
    # modern English, and unknown (word forms not expected and not recognized).
    # Can use Latin dictionaries/lexicons to discover what % over time the
    # corpus uses latin words. If possible, see if you can download pre-1800
    # historic English lexicon (to get old english stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok", help='directory containing corpus')
    parser.add_argument('--latin_dict', type=str, default="./latin_words.txt", help='text file containing latin dictionary')
    args = parser.parse_args()
    main(args)
