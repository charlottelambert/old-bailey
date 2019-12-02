#!/usr/bin/env python3
import os, argparse, csv, sys, copy, natsort, re
from tqdm import tqdm
from nltk.corpus import words
from flashtext import KeywordProcessor
import inflection as inf
from nltk.tag import pos_tag

english_words = KeywordProcessor()
latin_words = KeywordProcessor()

data_list = ["modern_english", "old_english", "latin", "proper_nouns", "upper", "lower", "mixed", "unk", "total"]


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

            # print("\n", len(all_tokens), "LINE:",' '.join(all_tokens))
            # Calculate capitalization statistics
            for word in all_tokens:
                if word.isupper():
                    stats_dict['upper'] += 1
                    # word = word.lower().capitalize()
                elif word.islower():
                    stats_dict['lower'] += 1
                else:
                    stats_dict['mixed'] += 1

            # propernouns: all proper nouns in the line
            # print("ALL", all_tokens)
            tagged_sent = pos_tag(all_tokens)
            tagged_nnp = [pair[0] for pair in tagged_sent if pair[1] == 'NNP']
            # print("nnp: ", tagged_nnp)
            # print("ALL", all_tokens)
            # propernouns = []
            # new_all = []
            # for i in range(len(all_tokens)):
            #     print(all_tokens[i])
            #     if all_tokens[i] in tagged_nnp:
            #         propernouns.append(all_tokens[i])
            #     else:
            #         new_all.append(all_tokens[i].lower()) # Lowercase because it isn't a proper noun anymore
            #         # print(all_tokens[i])
            # all_tokens = new_all
            all_tokens, propernouns = update_tok_lists(all_tokens, tagged_nnp)
            # print(len(all_tokens), "am i lower:",all_tokens)

            # propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
            # propernouns = [word for word in all_tokens if word in propernouns] # Make sure
            stats_dict['proper_nouns'] += len(propernouns)

            # english_words: all words in english in the line
            in_english = english_words.extract_keywords(" ".join(all_tokens))
            # english_words_found = []
            # for i, word in enumerate(all_tokens):
            #     if word in in_english:
            #         english_words_found.append(word)
            #         all_tokens.remove(word)
            all_tokens, english_words_found = update_tok_lists(all_tokens, in_english)

            stats_dict['modern_english'] += len(english_words_found)

            # latin_words: all latin words in the line
            in_latin = latin_words.extract_keywords(" ".join(all_tokens))
            # latin_words_found = []
            # for i, word in enumerate(all_tokens):
            #     if word in in_latin:
            #         latin_words_found.append(word)
            #         all_tokens.remove(word)
            all_tokens, latin_words_found = update_tok_lists(all_tokens, in_latin)
            stats_dict['latin'] += len(latin_words_found)


            stats_dict['unk'] += len(all_tokens)
            # # print("TAGS:", tagged_sent)
            # print(len(propernouns), "PROPER NOUNS:",propernouns)
            # # print(len(temporary), "TEMPORARY:", temporary)
            # print(len(english_words_found), "ENGLISH:",english_words_found)
            # print(len(latin_words_found), "LATIN:",latin_words_found)
            # print(len(all_tokens), "unk words:",all_tokens)
            # exit(0)
            try:
                assert stats_dict["modern_english"] + stats_dict["latin"] + stats_dict["old_english"] + stats_dict["proper_nouns"] + stats_dict["unk"] == stats_dict["total"]
                assert stats_dict["lower"] + stats_dict["upper"] + stats_dict["mixed"] == stats_dict["total"]
            except AssertionError:

                print(stats_dict)

                print("Error: failed to process file. Skipping", file, file=sys.stderr)
                # error_toks = []
                # for tok in all_tokens:
                #     if tok in propernouns or tok in english_words_found or tok in latin_words_found or tok in remaining_words or tok:
                #         continue
                #     temp = tok.lower()
                #     if temp in propernouns or temp in english_words_found or temp in latin_words_found or temp in remaining_words or tok:
                #         continue
                #     error_toks.append(tok)
                # print("Problematic tokens:", error_toks)

                # exit(1)
                return [backup, 0]
    # print(stats_dict)
    return [stats_dict, 1]

def init_stats_dict(data):
    stats_dict = {}
    for count in data:
        stats_dict[count] = 0
    return stats_dict

def get_order(file):
    base = os.path.basename(file)
    if base[:2] == "OA":
        return base[2:]
    return base

def main(args):
    # Add latin words to keyword processor
    with open(args.latin_dict) as f:
        latin_dict = f.read().split()
    latin_words.add_keywords_from_list(latin_dict)

    if args.english_words:
        with open(args.english_words) as f:
            ewords = f.read().split("\n")
        english_words.add_keywords_from_list(ewords)
    else:
        english_words.add_keywords_from_list(words.words())

    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]
    files = natsort.natsorted(files, key=lambda x: get_order(x))  # Sort in ascending numeric order
    # Initialize stats dict
    stats_dict = init_stats_dict(data_list)

    base_stats_dir = os.path.join(args.corpus_dir, "../stats_dir/")
    if not os.path.exists(base_stats_dir):
        os.mkdir(base_stats_dir)

    stats_path = os.path.join(base_stats_dir, args.corpus_dir.rstrip('/') + "_stats.tsv")
    with open(stats_path, "w") as f: # FIX OUTPUT DIRECTORY AND PATH
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(["start_year"] +  data_list)

        try:
            offset = 2 if os.path.basename(files[0])[:2] == "OA" else 0
            first_year = int(os.path.basename(files[0])[0 + offset:4 + offset]) # Find first year of earliest file
        except:
            print("Error: failed to process file. Skipping", files[0],  file=sys.stderr)

        year_idx = 0
        for i in tqdm(range(len(files))):
            file_path = files[i]
            offset = 2 if os.path.basename(file_path)[:2] == "OA" else 0
            try:
                year_idx = int(os.path.basename(file_path)[0+offset:4+offset]) - first_year
            except:
                print("Error: failed to process file. Skipping", file_path, file=sys.stderr)
                continue


            # If we've surpassed the time frame, write the row
            if year_idx >= args.year_split:
                # Write all data to tsv file (calculates over entire corpus)
                if valid:
                    tsv_writer.writerow([first_year] + [round(stats_dict[count]/stats_dict["total"], 4) for count in data_list])
                    print(stats_dict)


                stats_dict = init_stats_dict(data_list)
                first_year = int(os.path.basename(file_path)[0:4])

            stats_dict, valid = stats_for_file(file_path, stats_dict)

        if valid:
            tsv_writer.writerow([first_year] + [round(stats_dict[count]/stats_dict["total"], 4) for count in data_list])



    print("Wrote statistics to", stats_path, file=sys.stderr)
    # Estimate what amount of text is proper nouns, Latin, historical English,
    # modern English, and unknown (word forms not expected and not recognized).
    # Can use Latin dictionaries/lexicons to discover what % over time the
    # corpus uses latin words. If possible, see if you can download pre-1800
    # historic English lexicon (to get old english stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-stats", help='directory containing corpus')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to calculate stats for')
    parser.add_argument('--latin_dict', type=str, default="./latin_words.txt", help='text file containing latin dictionary')
    parser.add_argument('--english_words', type=str, default = "", help='optional path to file containing english words')
    parser.add_argument('--unique', default=False, action='store_true', help='whether or not to count only unique words')
    args = parser.parse_args()
    main(args)
