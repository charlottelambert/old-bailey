import re, nltk
from tqdm import tqdm

def parse_iterate(file, noun_counts=False, freq_dist=False):
    """
        Function to iterate over parsed file.

        input:
            file (string): filename of parsed file
            noun_counts (bool): whether or not to return number of proper
                nouns, non-proper nouns, and total words
            freq_dist (bool): Whether or not to return a frequency distribution
                of the words in a file
    """
    # Open input file and extract content
    with open(file, 'r') as f:
        content = f.read()
    matches = re.findall(r"\([A-Z]+ [a-z]+\)", content)

    if freq_dist: word_list = []

    # Initialize counts 
    if noun_counts:
        proper_nouns = 0
        other_nouns = 0
        total_words = len(matches)

    # Iterate over all words in corpus
    for match in matches:
        match = match[1:-1]
        pos, word = match.split()
        if freq_dist: word_list.append(word)
        if noun_counts:
            # If word is a noun, increment appropriate count
            if pos == 'NNP' or pos == "NNPS": proper_nouns += 1
            elif pos[0] == 'N': other_nouns += 1
    ret = []
    if noun_counts:
        ret += [proper_nouns, other_nouns, total_words]
    if freq_dist:
        ret += [nltk.FreqDist(word_list)]
    return ret

def noun_counts(file):
    """
        File must be in POS parsed format.
    """
    return parse_iterate(file, noun_counts=True)

def bnc_fd(files):
    """
        Function to generate frequency distribution for words in BNC files.

        input:
            files (list): list of filenames that coprise the BNC

        returns type nltk.freqdist object of all the words in BNC
    """
    fd = nltk.FreqDist()
    for i in tqdm(range(len(files))):
        file = files[i]
        fd.update(parse_iterate(file, freq_dist=True)[0])
    return fd
