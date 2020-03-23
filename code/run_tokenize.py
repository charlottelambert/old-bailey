#!/usr/bin/env python3

###############################################################################
# run-tokenize.py
#
# Tokenize text using word_tokenize and replace split-up contractions with
# equivalent words (i.e., couldn't -> could not). Should be run after text
# is dehyphenated.
#
###############################################################################

import sys, argparse, os, re, enchant, json, nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from utils import *
from nltk.corpus import stopwords

stop_words = []

def contractions(token_list):
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

def tokenize_file(args, file, output_dir, gb, gb_and_pwl, bigrams):
    output = []
    with open(file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            # Lower line if needed
            if args.lower and not args.stats:
                line = line.lower()

            # Tokenize line
            tokens = [word.replace("\\", "") for word in word_tokenize(line)]

            # Join $ to names
            for i, tok in enumerate(tokens):
                if tok == '$':
                    tokens[i:i+2] = [''.join(tokens[i:i+2])]

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

            # Turn into bigrams if flag is true
            if args.bigrams and not args.stats:
                tokens = make_bigrams(tokens)
                output.append(" ".join(tokens))
                continue

            # Handle issue with dashes appearing at start of word
            mod_tokens = []
            for i in range(len(tokens)):
                mod_tokens += tokens[i].split() #fix_hyphens(tokens[i])
            tokens = mod_tokens

            # Replace split contractions with full words
            tokens = contractions(tokens)

            if not args.stats:
                # First, remove trailing hyphens and slashes
                sub_pattern = '\A([\W_]*)([A-Za-z0-9]+|[A-Za-z0-9]+[\W_]+[A-Za-z0-9]+)([\W_]*)$'
                tokens = [re.sub(sub_pattern, "\\2", x) for x in tokens]

                # Keep all words containing at least one letter
                # Also remove words of length < 2
                # Lemmatize if necessary
                if args.lemma:
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(x) for x in tokens
                              if len(x) > 2 and x.lower() not in stop_words
                              and re.search('[a-zA-Z]', x)]
                else:
                    tokens = [x for x in tokens if len(x) > 2
                              and x.lower() not in stop_words
                              and re.search('[a-zA-Z]', x)]

            # If tokenizing text in order to find useful stats, do extra
            # processing and return without removing words
            if args.stats:
                output.append(" ".join(tokens))
                continue

            finished = " ".join(tokens)

            # If needed, replace street names with generic version
            if args.street_sub:
                finished = re.sub("([^ ]+\-street)|([A-Z][a-z]* street)", "$name_street", finished)
                # finished = re.sub("([^ ]+\-lane)|([A-Z][a-z]* lane)", "$name_street", finished)
                # finished = re.sub("([^ ]+\-road)|([A-Z][a-z]* road)", "$name_street", finished)
                # finished = re.sub("[^ ]+\-row", "$name_street", finished)
                # also -square, -highway, -cross, -grove, -town

            output.append(finished)
    return output

# Eventually, we jsut want to run this on each word, might need a way to speed
# this up, parallel?
# gonna have millions of tokens, need to be more efficient!
def spell_correct(args, gb, gb_and_pwl, word, bigrams):
    # If the line is a valid word, continue
    if word == "" or word[0].isupper() or gb.check(word):
        # print("Is a word:", word)
        return word
    else:
        # print("not a word:", word)
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
                if bigrams[opt] > best[1]:
                    best = (opt, bigrams[opt])
            except KeyError:
                continue

    return best[0]

def merge_words(args, pwl, input, bigrams):
    """
        Go through the text and for every pair of words, check if they're in
        the unigram list (args.pwl_path) when you remove the space. If so, make
        the change.
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

    if not args.disable_stopwords:
        stop_words = set(stopwords.words('english'))

    if args.test:
        print("starting test")
        enchant.set_param("enchant.myspell.dictionary.path", args.myspell_path)
        gb = enchant.DictWithPWL("en_GB") #, args.pwl_path) # GB isn't working, doesn't recognize 'entrancei' as "entrance i"
        gb_and_pwl = enchant.DictWithPWL("en_GB", args.pwl_path) # GB isn't working, doesn't recognize 'entrancei' as "entrance i"
        change_set = set()
        with open(args.filepath) as f:
            lines = f.read().split()
            # for i in tqdm(range(len(lines))):
                # line = lines[i].rstrip()
            for line in lines:
                tokens = [word.replace("\\", "") for word in word_tokenize(line.rstrip())]

                # Join $ to names
                for i, tok in enumerate(tokens):
                    if tok == '$':
                        tokens[i:i+2] = [''.join(tokens[i:i+2])]

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

                        if spell_checked[0] != sub:
                            change_set.add(sub + ": " + " ".join(spell_checked))

        print("\n".join(change_set))
        exit(0)

    # Define additional info to add to output path
    suffix = "-tok"
    sp_str = "-no_sp" if args.disable_spell_check else ""
    bigram_str = "-bi" if args.bigrams else ""
    lower_str = "-lower" if args.lower else ""
    lemma_str = "-lemma" if args.lemma else ""
    street_str = "-streets" if args.street_sub else ""
    stop_str = "-tng" if args.disable_stopwords else "" # tng = topical ngrams

    suffix += sp_str + bigram_str + lower_str + lemma_str + street_str + stop_str

    if args.stats:
        suffix = "-stats"


    if not args.output_dir_base:
        base = args.corpus_dir if not args.filepath else os.path.dirname(args.filepath)
        output_dir = base.rstrip("/") + suffix
    else:
        output_dir = args.output_dir_base.rstrip("/") + suffix
    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not args.filepath:
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
        output = tokenize_file(args, args.filepath, output_dir, gb, gb_and_pwl, bigrams)
        # Merge words if flag is set to true
        if args.merge_words:
            # Create dictionary (personal word list) out of unigrams
            pwl = enchant.request_pwl_dict(args.pwl_path)

            for i,line in enumerate(output):
                output[i] = " ".join(merge_words(args, pwl, line.split(), bigrams))

        # Write output to new file
        with open(output_file, "w") as f:
            f.write("\n".join(output))
        exit(0)
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
            output = tokenize_file(args, file, output_dir, gb, gb_and_pwl, bigrams)
            # Write output to new file
            with open(output_file, "w") as f:
                f.write('\n'.join(output))
        print(timestamp() + " Tokenization done.", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_base', type=str, default="", help='location to save tokenized text')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt", help='directory containing corpus')
    parser.add_argument('--filepath', type=str, default="", help='path to single file to be tokenized')
    parser.add_argument('--overwrite', default=False, action="store_true", help='whether or not to overwrite old files with the same names')
    parser.add_argument('--bigrams', default=False, action="store_true", help='whether or not to convert data to bigrams')
    parser.add_argument('--lower', default=False, action="store_true", help='whether or not to lowercase all text')
    parser.add_argument('--stats', default=False, action="store_true", help='whether or not to process text for finding statistics (calc_stats.py)')
    parser.add_argument('--street_sub', default=False, action="store_true", help='whether or not to substitute street names with generic string')
    parser.add_argument('--lemma', default=False, action="store_true", help='whether or not to lemmatize all text')
    parser.add_argument('--test', default=False, action="store_true", help='testing spellcheck')
    parser.add_argument('--corpus_bigrams', type=str, default="/work/clambert/thesis-data/OB_LL-txt/corpus_bigrams.json", help='path to json file containing dictionary of all corpus bigrams')
    parser.add_argument('--myspell_path', type=str, default="/home/clambert/.local/lib/python3.6/site-packages/enchant/share/enchant/myspell")
    parser.add_argument('--disable_spell_check', default=False, action="store_true", help='whether or not to disable spell check')
    parser.add_argument('--pwl_path', type=str, default="/work/clambert/thesis-data/OB_LL-txt/unigram_pwl.txt")
    parser.add_argument('--merge_words', default=False, action="store_true", help='whether or not to merge words into words present in unigram list')
    parser.add_argument('--disable_stopwords', default=False, action="store_true", help='whether or not to disable stop word removal')
    # /System/Volumes/Data/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/enchant/share/enchant/myspell
    args = parser.parse_args()
    main(args)
