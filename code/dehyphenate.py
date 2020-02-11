#!/usr/bin/env python3

###############################################################################
# dehyphenate.py
#
# Dehyphenate one file
#
###############################################################################
import sys, argparse, os, re
from tqdm import tqdm
# from nltk.corpus import words
from flashtext import KeywordProcessor
import inflection as inf

keyword_processor = KeywordProcessor()
def dehyphenate(token):
    """
        Dehyphenate input token. First, try replacing hyphen with a space. If
        it yields two or more valid words, return that. Next, try replacing
        hyphen with the empty string. If that yields one valid word, return
        that. Otherwise, return word as is, hyphens are assumed to be
        appropriate.

        token: string to dehyphenate
    """
    if(not "—" in token and not "-" in token) or token[:1] == "$":
        return token

    if "—" in token:
        split_tok = "—"
    else:
        split_tok = "-"

    # # If bigrams, run over each unigram
    # if "_" in token:
    #     words = token.split("_")
    #     output = []
    #     for word in words:
    #         new_word = dehyphenate(word)
    #         output.append(new_word)
    #     return "_".join(output)

    options = [" ", ""]
    for opt in options:
        ready = True
        mod_tokens = opt.join(token.split(split_tok))
        # Look for all words in the list of tokens
        words_found = keyword_processor.extract_keywords(mod_tokens)

        # If all tokens were words, return new tokens
        if len(words_found) == len(mod_tokens.split()):
            return mod_tokens

        # Check if all tokens are words if they are singularized
        # (words.words() does not contain plural words)
        singular_words_found = keyword_processor.extract_keywords(" ".join([inf.singularize(tok) for tok in mod_tokens.split()]))
        if len(singular_words_found) == len(mod_tokens.split()):
            return mod_tokens

    return token

def main(args):
    # Define output directory (if not provided)
    output_dir = os.path.dirname(args.file).rstrip("/") + "-dh"

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(args.english_words) as f:
        content = f.read().split("\n")
        keyword_processor.add_keywords_from_list(content)

    output_file = os.path.join(output_dir, os.path.basename(args.file))
    if not args.overwrite and os.path.exists(output_file):
        print("File", output_file, "exists. Skipping...")

    output = []
    with open(args.file, "r") as f:
        for line in f:
            tokens = [dehyphenate(tok) for tok in line.split()]
            output.append(" ".join(tokens))

    with open(output_file, "w+") as f:
        f.write('\n'.join(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='file to dehyphenate')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--english_words', type=str, default = "/work/clambert/thesis-data/bnc_lexicon.txt", help='optional path to file containing english words')
    args = parser.parse_args()
    main(args)
