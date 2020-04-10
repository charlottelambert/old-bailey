import re, nltk
from tqdm import tqdm

def parse_iterate(file, noun_counts=False, freq_dist=False):
    # Open input file and extract content
    with open(file, 'r') as f:
        content = f.read()
    matches = re.findall(r"\([A-Z]+ [a-z]+\)", content)

    if freq_dist: word_list = []
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
    fd = nltk.FreqDist()
    for i in tqdm(range(len(files))):
        file = files[i]
        fd.update(parse_iterate(file, freq_dist=True)[0])
    return fd
