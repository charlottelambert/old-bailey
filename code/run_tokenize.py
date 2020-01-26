#!/usr/bin/env python3

###############################################################################
# run-tokenize.py
#
# Tokenize text using word_tokenize and replace split-up contractions with
# equivalent words (i.e., couldn't -> could not). Should be run after text
# is dehyphenated.
#
###############################################################################

import sys, argparse, os, re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

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

def make_bigrams(tokens):
    if len(tokens) == 0:
        return tokens
    # If using bigrams, convert tokenized unigrams to bigrams
    bigrams = list(ngrams(tokens, 2))
    output = []
    for bigram in bigrams:
        output.append(bigram[0] + "_" + bigram[1])
    return output

def fix_hyphens(input):
    """
        Fix issues with hyphens that isn't fixed through tokenization.
    """
    out = re.sub(r'([^a-zA-z-])([\—\-]+)([^a-zA-z-])', '\\1 \\2 \\3', input)
    out = re.sub(r'([^a-zA-z-])([\—\-]+)([a-zA-z])', '\\1 \\2 \\3', out)
    out = re.sub(r'([a-zA-z])([\—\-]+)([^a-zA-z-])', '\\1 \\2 \\3', out)
    out = re.sub(r'([a-zA-z])([\—\-]+)([a-zA-z-])', '\\1 \\2 \\3', out)

    return out.split()

def main(args):
    # Compile list of files to tokenize
    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]

    # Define additional info to add to output path
    suffix = "-tok"
    bigram_str = "-bi" if args.bigrams else ""
    lower_str = "-lower" if args.lower else ""
    street_str = "-streets" if args.street_sub else ""

    suffix += bigram_str + lower_str + street_str

    if args.stats:
        suffix = "-stats"

    # Define output directory (if not provided)
    if not args.output_dir_base:
        output_dir = args.corpus_dir.rstrip("/") + suffix
    else: output_dir = args.output_dir_base.rstrip("/") + suffix

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
                if not line.strip():
                    continue

                # Lower line if needed
                if args.lower and not args.stats:
                    line = line.lower()

                # Fix apostrophe escapes:
                line = line.replace("\'", "'")

                # Tokenize line
                tokens = word_tokenize(line)

                # Join $ to names
                for i, tok in enumerate(tokens):
                    if tok == '$':
                        tokens[i:i+2] = [''.join(tokens[i:i+2])]

                if args.bigrams and not args.stats:
                    tokens = make_bigrams(tokens)
                    output.append(" ".join(tokens))
                    continue

                # Handle issue with dashes appearing at start of word
                mod_tokens = []
                for i in range(len(tokens)):
                    mod_tokens += fix_hyphens(tokens[i])
                tokens = mod_tokens

                if not args.stats:
                    # Keep all words containing at least one letter
                    tokens = [x for x in tokens if re.search('[a-zA-Z]', x)]

                # Replace split contractions with full words
                tokens = contractions(tokens)

                # If tokenizing text in order to find useful stats, do extra
                # processing and return without removing words
                if args.stats:
                    # Get rid of problem with slashes
                    output.append(" ".join(tokens))
                    continue

                # Remove words of length < 2
                tokens = [x for x in tokens if len(x) > 2]
                # Remove trailing slashes
                tokens = [re.sub(r'([^\\]*)(\\+)', '\\1', x) for x in tokens]
                finished = " ".join(tokens)

                # If needed, replace street names with generic version
                if args.street_sub:
                    finished = re.sub("([^ ]+\-street)|([A-Z][a-z]* street)", "$name_street", finished)
                    # finished = re.sub("([^ ]+\-lane)|([A-Z][a-z]* lane)", "$name_street", finished)
                    # finished = re.sub("([^ ]+\-road)|([A-Z][a-z]* road)", "$name_street", finished)
                    # finished = re.sub("[^ ]+\-row", "$name_street", finished)
                    # also -square, -highway, -cross, -grove, -town

                output.append(finished)
        with open(output_file, "w+") as f:
            f.write('\n'.join(output))
    print("Tokenization done.", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_base', type=str, default="", help='location to save tokenized text')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt", help='directory containing corpus')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--bigrams', default=False, action="store_true", help='whether or not to convert data to bigrams')
    parser.add_argument('--lower', default=False, action="store_true", help='whether or not to lowercase all text')
    parser.add_argument('--stats', default=False, action="store_true", help='whether or not to process text for finding statistics (calc_stats.py)')
    # parser.add_argument('--no_proper_nouns', default=False, action="store_true", help='whether or not to discard all proper nouns')
    parser.add_argument('--street_sub', default=False, action="store_true", help='whether or not to substitute street names with generic string')
    args = parser.parse_args()
    main(args)
