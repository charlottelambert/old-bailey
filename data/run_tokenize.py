#!/usr/bin/env python3

###############################################################################
# run-tokenize.py
#
# Tokenize text using word_tokenize and replace split-up contractions with
# equivalent words (i.e., couldn't -> could not).
#
###############################################################################

import sys, argparse, os, re, enchant, json, nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import stopwords
sys.path.append('../')
from utils import *

stop_words = set(stopwords.words('english'))

def contractions(token_list):
    """
        Function to replace split up contractions with two full words
        (e.g., "do n't" --> "do not").

        input:
            token_list (list): list of tokens

        returns token list with replaced contractions.
    """
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
    """
        Turn tokens into bigrams.

        input:
            tokens (list): list of tokens to convert

        returns list of bigrams built from input tokens.
    """
    if len(tokens) == 0:
        return tokens
    # If using bigrams, convert tokenized unigrams to bigrams
    bigrams = list(ngrams(tokens, 2))
    output = []
    for bigram in bigrams:
        output.append(bigram[0] + "_" + bigram[1])
    return output

def remove_unwanted(args, tokens):
    """
        Remove unwanted words from tokens including words that are too short
        and stopwords (if not disabled). Also lemmatize if flag is passed in.

        input:
            args (argparse object): input arguments
            tokens (list): list of tokens

        return list of tokens with unwanted ones removed.
    """
    if args.disable_stopwords: s = []
    else: s = stop_words

    if args.lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(x) for x in tokens
                  if len(x) > 2 and x.lower() not in s
                  and re.search('[a-zA-Z]', x)]
    else:
        tokens = [x for x in tokens if len(x) > 2
                  and x.lower() not in s
                  and re.search('[a-zA-Z]', x)]
    return tokens

def tokenize_line(args, line, gb, gb_and_pwl, bigrams):
    """
        Function to tokenize one line of a file.

        input:
            args (arparse object): input arguments
            line (str): line from a file
            gb: british dictionary for spell checking
            gb_and_pwl: words from british dictionary and input personal word
                list
            bigrams (dict): corpus bigrams read from args.corpus_bigrams

        returns tokenized line joined back into a string (to be written to file)
    """
    if not line.strip():
        return ""

    # Lower line if needed
    if args.lower:
        line = line.lower()

    # Tokenize line
    tokens = [word.replace("\\", "") for word in word_tokenize(line)]

    # Spelling correction
    updated_tokens = []
    for i, tok in enumerate(tokens):
        # Split word by non-alphanumeric characters
        split_word = re.split("([^A-Za-z0-9_(\w'\w)])|(^')|('$)", tok)

        split_word = [w for w in split_word if not w == None and len(w) > 2]
        if args.disable_spell_check:
            updated_tokens += split_word
            continue

        for sub in split_word:
            spell_checked = spell_correct(args, gb, gb_and_pwl, sub, bigrams).split()
            updated_tokens += spell_checked
    tokens = updated_tokens
    # Handle issue with dashes appearing at start of word
    mod_tokens = []
    for i in range(len(tokens)):
        mod_tokens += tokens[i].split()
    tokens = mod_tokens

    # Replace split contractions with full words
    tokens = contractions(tokens)

    # First, remove trailing hyphens and slashes
    sub_pattern = '\A([\W_]*)([A-Za-z0-9]+|[A-Za-z0-9]+[\W_]+[A-Za-z0-9]+)([\W_]*)$'
    tokens = [re.sub(sub_pattern, "\\2", x) for x in tokens]
    tokens = remove_unwanted(args, tokens)

    for idx, t in enumerate(tokens):
        tokens[idx] = "$" + t if "_" in t else t

    # Turn into bigrams if flag is true
    if args.bigrams:
        tokens = make_bigrams(tokens)


    finished = " ".join(tokens)

    # If needed, replace street names with generic version
    if args.street_sub:
        finished = re.sub("([^ ]+\-street)|([A-Z][a-z]* street)", "$name_street", finished)
        # finished = re.sub("([^ ]+\-lane)|([A-Z][a-z]* lane)", "$name_street", finished)
        # finished = re.sub("([^ ]+\-road)|([A-Z][a-z]* road)", "$name_street", finished)
        # finished = re.sub("[^ ]+\-row", "$name_street", finished)
        # also -square, -highway, -cross, -grove, -town

    return finished

def tokenize_file(args, file, gb, gb_and_pwl, bigrams):
    """
        Function to tokenize each line in a file.

        input:
            args (argparse object): input arguments
            file (str): path to file to tokenize
            gb: british dictionary for spell checking
            gb_and_pwl: words from british dictionary and input personal word
                list
            bigrams (dict): corpus bigrams read from args.corpus_bigrams

        returns list of tokenized lines
    """
    output = []
    with open(file, "r") as f:
        for line in f:
            output.append(tokenize_line(args, line, gb, gb_and_pwl, bigrams))
    return output


def spell_correct(args, gb, gb_and_pwl, word, bigrams):
    """
        Function to spell-check a word and correct it if possible.

        input:
            args (argparse object): input arguments
            gb: british dictionary for spell checking
            gb_and_pwl: words from british dictionary and input personal word
                list
            word (str): word to spell-check
            bigrams (dict): corpus bigrams read from args.corpus_bigrams

        returns spell-checked (and corrected, if necessary) word
    """
    # If the line is a valid word, continue
    if word == "" or word[0].isupper() or gb.check(word): return word
    else:
        # Suggest corrections for sub_line
        suggestions = gb_and_pwl.suggest(word)
        # See if any of them are reasonable
        options = []
        for opt in suggestions:
            l = opt.split()
            if len(l) == 2 and "".join(l) == word:
                options.append(opt)
                break
        # Find the most probable option
        best = (word, 0)
        for opt in options:
            try:
                # Check if option is a bigram that appears in corpus
                if bigrams[opt] > best[1]: best = (opt, bigrams[opt])
            except KeyError: continue

    return best[0]

def merge_words(args, pwl, input, bigrams):
    """
        Go through the text and for every pair of words, check if they're in
        the unigram list (args.pwl_path) when you remove the space. If so, make
        the change.

        input:
            args (argparse object): input arguments
            pwl: personal word list loaded from args.pwl_path
            input: line of file to merge words in (if necessary)
            bigrams (dict): corpus bigrams read from args.corpus_bigrams

        returns list of words from input, merged if necessary
    """
    output = []

    # Compile list of bigrams
    bg = nltk.bigrams(input)
    skip = False
    store = ""
    for b in bg:
        # Indicates last bigram was merged, don't want to consider this bigram
        if skip or (b[0] == '' or b[1] == ''): # or b in bigrams
            skip = False
            continue
        merged = "".join(b)
        if pwl.check(merged):
            output.append(merged)
            skip = True
        else:
            output.append(b[0])
            store = b[1]
    # If we ended on a non-valid merge, append the last unigram
    if not skip:
        output.append(store)

    return output

def main(args):
    with open(args.corpus_bigrams) as json_file:
        bigrams = json.load(json_file)

    # Define additional info to add to output path
    suffix = "-tok"
    sp_str = "-no_sp" if args.disable_spell_check else ""
    bigram_str = "-bi" if args.bigrams else ""
    lower_str = "-lower" if args.lower else ""
    lemma_str = "-lemma" if args.lemma else ""
    street_str = "-streets" if args.street_sub else ""
    stop_str = "-tng" if args.disable_stopwords else "" # tng = topical ngrams

    suffix += sp_str + bigram_str + lower_str + lemma_str + street_str + stop_str

    if not args.output_dir_base:
        base = args.corpus_dir if not args.filepath else os.path.dirname(args.filepath)
        output_dir = base.rstrip("/") + suffix
    else:
        output_dir = args.output_dir_base.rstrip("/") + suffix

    # Create output directory
    if not args.tsv_corpus and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not args.filepath and not args.tsv_corpus:
        print(timestamp() + " Tokenizing data to", suffix, file=sys.stderr)

    enchant.set_param("enchant.myspell.dictionary.path", args.myspell_path)
    gb = enchant.DictWithPWL("en_GB") #, args.pwl_path) # GB isn't working, doesn't recognize 'entrancei' as "entrance i"
    gb_and_pwl = enchant.DictWithPWL("en_GB", args.pwl_path) # GB isn't working, doesn't recognize 'entrancei' as "entrance i"

    # If processing one file, don't loop!
    if args.filepath:
        if not os.path.splitext(args.filepath)[1] == ".txt":
            print(timestamp() + " Must input text file. Exiting...", file=sys.stderr)
            exit(0)
        output_file = os.path.join(output_dir, os.path.basename(args.filepath))
        if not args.overwrite and os.path.exists(output_file):
            exit(0)
        # Tokenize single file
        output = tokenize_file(args, args.filepath, gb, gb_and_pwl, bigrams)
        # Merge words if flag is set to true
        if args.merge_words:
            # Create dictionary (personal word list) out of unigrams
            pwl = enchant.request_pwl_dict(args.pwl_path)

            for i,line in enumerate(output):
                output[i] = " ".join(merge_words(args, pwl, line.split(), bigrams))

        if not os.path.exists(os.path.dirname(output_file)): os.makedirs(os.path.dirname(output_file))
        # Write output to new file
        with open(output_file, "w") as f:
            f.write("\n".join(output))
        exit(0)
    else:
        if args.tsv_corpus:
            output_file = args.tsv_corpus[:-4] + suffix + ".tsv"
            if not args.overwrite and os.path.exists(output_file):
                print("File", output_file, "exists. Exiting...")
                exit(0)
            with open(args.tsv_corpus, 'r') as f:
                docs = f.read().split("\n")
                if docs[0].lower() == "id\tyear\ttext":
                    idx = 1
                    tsv_out = [docs[0]]
                else:
                    idx = 0
                    tsv_out = []
                for doc in docs[idx:]:
                    try:
                        id, year, text = doc.split("\t")
                    except ValueError: continue
                    tokenized = tokenize_line(args, text, gb, gb_and_pwl, bigrams)
                    tsv_out.append(id + "\t" + year + "\t" + tokenized)
            with open(output_file, "w") as f:
                f.write('\n'.join(tsv_out))
        else:
            # Compile list of files to tokenize
            files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
                     if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]

            for i in tqdm(range(len(files))):
                file = files[i]
                # Define path for new tokenized file
                output_file = os.path.join(output_dir, os.path.basename(file))
                if not args.overwrite and os.path.exists(output_file):
                    continue

                # Tokenize single file
                output = tokenize_file(args, file, gb, gb_and_pwl, bigrams)

                # Write output to new file
                with open(output_file, "w") as f:
                    f.write('\n'.join(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_corpus', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt.tsv", help='path to data in tsv format')
    parser.add_argument('--output_dir_base', type=str, default="", help='location to save tokenized text')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt", help='directory containing corpus')
    parser.add_argument('--filepath', type=str, default="", help='path to single file to be tokenized')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--bigrams', default=False, action="store_true", help='whether or not to convert data to bigrams')
    parser.add_argument('--lower', default=False, action="store_true", help='whether or not to lowercase all text')
    parser.add_argument('--street_sub', default=False, action="store_true", help='whether or not to substitute street names with generic string')
    parser.add_argument('--lemma', default=False, action="store_true", help='whether or not to lemmatize all text')
    parser.add_argument('--corpus_bigrams', type=str, default="/work/clambert/thesis-data/OB_LL-txt/corpus_bigrams.json", help='path to json file containing dictionary of all corpus bigrams (./ngrams.py)')
    parser.add_argument('--myspell_path', type=str, default="/home/clambert/.local/lib/python3.6/site-packages/enchant/share/enchant/myspell", help='path to myspell dictionary')
    parser.add_argument('--disable_spell_check', default=False, action="store_true", help='whether or not to disable spell check')
    parser.add_argument('--pwl_path', type=str, default="/work/clambert/thesis-data/OB_LL-txt/unigram_pwl.txt", help='path to unigram word list')
    parser.add_argument('--merge_words', default=False, action="store_true", help='whether or not to merge words into words present in unigram list')
    parser.add_argument('--disable_stopwords', default=False, action="store_true", help='whether or not to disable stop word removal')
    args = parser.parse_args()
    main(args)
