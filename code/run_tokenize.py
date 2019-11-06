#!/usr/bin/env python3

###############################################################################
# run-tokenize.py
#
# Tokenize text using word_tokenize and replace split-up contractions with
# equivalent words (i.e., couldn't -> could not)
#
###############################################################################

import sys, argparse, os, re
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def contractions(token_list):
    # Replace contraction tokenization with real words
    c_dict =    {
    "ca n't": "can not",
    "wo n't": "will not",
    "couldn't 've": "could not have",
    "could n't": "could not",
    "n't": "not",
    "'d": "would",
    "'ll": "will",
    "'ve": "have",
    "'m": "am",
    "let 's": "let us",
    "'re": "are"
    }

    # Remove all asterisks and replace contractions
    for idx, token in enumerate(token_list):
        if token == "*":
            token_list[idx] = ""
        elif token in c_dict:
            token_list[idx] = c_dict[token]
        elif idx < len(token_list) - 1:
            combined = token + " " + token_list[idx + 1]
            if combined in c_dict:
                token_list[idx] = c_dict[combined]
                token_list[idx+1] = ""
    return token_list


def fix_hyphens(input):
    """
        Fix issues with hyphens that isn't fixed through tokenization.
    """
    out = re.sub(r'([^a-zA-z-])(\-+)([^a-zA-z-])', '\\1 \\2 \\3', input)
    out = re.sub(r'([^a-zA-z-])(\-+)([a-zA-z])', '\\1 \\2 \\3', out)
    out = re.sub(r'([a-zA-z])(\-+)([^a-zA-z-])', '\\1 \\2 \\3', out)
    return out

def main(args):
    # Compile list of files to tokenize
    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]

    # Define output directory (if not provided)
    if not args.output_dir_base:
        output_dir = args.corpus_dir.rstrip("/") + "-tok"
    else: output_dir = args.output_dir_base.rstrip("/") + "-tok"

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(len(files))):
        file = files[i]
        output_file = os.path.join(output_dir, os.path.basename(file))
        if not args.overwrite and os.path.exists(output_file):
            continue

        output = []
        with open(file, "r") as f:
            for line in f:
                # Tokenize line
                tokens = word_tokenize(line)
                # Handle issue with dashes appearing at start of word
                # tokens = [[s for s in fix_hyphens(token)][0] for token in tokens]
                tokens = [fix_hyphens(token) for token in tokens]
                # Keep all words containing at least one letter
                tokens = [x for x in tokens if re.search('[a-zA-Z]', x)]
                # Replace split contractions with full words
                tokens = contractions(tokens)
                # Remove words of length < 2
                tokens = [x for x in tokens if len(x) > 2]
                output.append(" ".join(tokens))
        with open(output_file, "w+") as f:
            f.write('\n'.join(output))
    print("Tokenization done.", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_base', type=str, default="", help='location to save tokenized text')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt", help='directory containing corpus')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    args = parser.parse_args()
    main(args)
