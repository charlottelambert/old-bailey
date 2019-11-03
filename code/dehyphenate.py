#!/usr/bin/env python3

###############################################################################
# dehyphenate.py
#
# Dehyphenate one file
#
###############################################################################
import sys, argparse, os, re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import words

def dehyphenate(token):
    """
        Dehyphenate input token. First, try replacing hyphen with a space. If
        it yields two or more valid words, return that. Next, try replacing
        hyphen with the empty string. If that yields one valid word, return
        that. Otherwise, return word as is, hyphens are assumed to be
        appropriate.

        token: string to dehyphenate
    """
    if not "-" in token:
        return token

    options = [" ", ""]
    for opt in options:
        ready = True
        mod_tokens = opt.join(token.split("-"))
        for tok in mod_tokens.split():

            # check if tok is a word
            if not tok in words.words():
                ready = False
                break
        if ready:
            return mod_tokens
    return token

def main(args):
    # Define output directory (if not provided)
    output_dir = os.path.dirname(args.file).rstrip("/") + "-dh"

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


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
    args = parser.parse_args()
    main(args)
