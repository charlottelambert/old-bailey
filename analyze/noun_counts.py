import re

def noun_counts(file):
    proper_nouns = 0
    other_nouns = 0

    # Open input file and extract content
    with open(file, 'r') as f:
        content = f.read()
    matches = re.findall(r"\([A-Z]+ [a-z]+\)", content)
    total_words = len(matches)

    # Iterate over all words in corpus
    for match in matches:
        match = match[1:-1]
        pos, word = match.split()

        # If word is a noun, increment appropriate count
        if pos == 'NNP': proper_nouns += 1
        elif pos[0] == 'N': other_nouns += 1

    return proper_nouns, other_nouns, total_words
