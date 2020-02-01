#!/usr/bin/env python3
import sys, re, argparse, os
from tqdm import tqdm
from bs4 import BeautifulSoup
from run_tokenize import fix_hyphens

# Extract words from one XML file and compile
def update_bnc_words(xml_path):
    with open(xml_path) as f:
        content = f.read()

    xml_content = BeautifulSoup(content, features="lxml")
    xml_content = xml_content.text.split()

    bnc_words = set()
    for raw_word in xml_content:
        # Remove trailing single quotes
        raw_word = re.sub(r'(.+)\'\s+', "\\1", raw_word)

        # Split on slashes
        word_list = raw_word.split("/")

        # Add all actual words to the list
        for word in word_list:
            fixed_word = fix_hyphens(word)
            # May be a proper noun, add both lower and upper versions
            is_word = re.match('[a-zA-Z]', fixed_word)

            if is_word:
                bnc_words.add(fixed_word.lower())

    return bnc_words

def main(args):
    bnc_words = set()

    files = [os.path.join(args.xml_base_path, f) for f in os.listdir(args.xml_base_path)
             if (os.path.isfile(os.path.join(args.xml_base_path, f)) and f.endswith('.xml'))]
    # Iterate over each file in the sub directories
    for i in tqdm(range(len(files))):
        xml_path = files[i]
        # Extract the words from that path
        bnc_words.update(update_bnc_words(xml_path))

    # Create directory for output file if doesn't exist
    if not os.path.exists(os.path.dirname(args.save_lexicon_path)):
        os.makedirs(os.path.dirname(args.save_lexicon_path))

    # Save compiled list of words to a path
    with open(args.save_lexicon_path, "w") as f: # FIX OUTPUT DIRECTORY AND PATH
        f.write("\n".join(bnc_words))
        print("wrote to", args.save_lexicon_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_base_path', type=str, default="/work/jgordon/tmp/bnc/text", help='base directory for all BNC xml files')
    parser.add_argument('--save_lexicon_path', type=str, default="/work/clambert/thesis-data/bnc_lexicon.txt", help='output file path')
    args = parser.parse_args()
    main(args)
