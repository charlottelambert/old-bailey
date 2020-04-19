#!/usr/bin/env python3
import os, argparse, natsort, sys, re
from datetime import datetime

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

def timestamp():
    """
        Function to return formatted time string.
    """
    return "["+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"]"

def get_order(file):
    base = os.path.basename(file)
    if base[:2] == "OA":
        return base[2:]
    if base[:1].isalpha():
        return base[1:]
    return base

def get_year(input, include_month=False, tsv=False):
    """
        Get the year a file refers to.

        file: input filepath of format "OAYYYYMMDD.txt" or "YYYYMMDD.txt" or
                    "XYYYYMMDD.txt" where X is a single lowercase letter.

        returns year in int format
    """
    if tsv:
        cols = input.split("\t")
        if cols[0].lower() == "id": return -1
        return int(cols[1])
    try:
        if os.path.basename(input)[:2] == "OA": offset = 2
        elif os.path.basename(input)[:1].isalpha(): offset = 1
        else: offset = 0
        year = int(os.path.basename(input)[0 + offset:4 + offset])
        if include_month:
            month = int(os.path.basename(input)[4 + offset:6 + offset])
            return (year, month)
        return year
    except ValueError:
        print(timestamp() + " Skipping invalid file", input, file=sys.stderr)
        return -1

def order_files(args):
    """
        Sort a list of input files by years.

        input: args
        output: dictionary of format {"YYYY":[file0,file1,...], "YYYY+args.year_split":[file0,file1,...]}
                and time slices:: [a, b, c]
    """
    # If given tsv file as input, order documents based on year
    tsv = False
    try:
        with open(args.corpus_file, 'r') as f:
            lines = f.read().split("\n")
            lines = [line for line in lines if line.rstrip()]
            tsv = True
        # If there was an input london lives tsv file, add that to the documents
        try:
            with open(args.london_lives_file, 'r') as f:
                ll_lines = f.read().split("\n")
                ll_lines = [line for line in ll_lines if line.rstrip()]
            lines += ll_lines
        except: pass

        docs = natsort.natsorted(lines, key=lambda x: x.split("\t")[1])

    # Otherwise, input is directory of files
    except:
        docs = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
                 if (os.path.isfile(os.path.join(args.corpus_dir, f))
                     and re.match(".*[0-9]{8}", f) and f.endswith('.txt'))]

        docs = natsort.natsorted(docs, key=lambda x: get_order(x))  # Sort in ascending numeric order

    # Find start year
    start_year = get_year(docs[0], tsv=tsv)
    docs_dict = {start_year:[]}

    # Determine if we want to split by year
    try:
        year_split = args.year_split
    except AttributeError:
        year_split = -1

    # If no split, just return docs as is
    if year_split == -1:
        docs_dict[start_year] = docs
        return [docs_dict, len(docs)]

    for doc in docs:
        # Get year for current document
        cur_year = get_year(doc, tsv=tsv)
        if cur_year == -1: continue
        if cur_year - start_year >= args.year_split:
            start_year = cur_year
            docs_dict[start_year] = []
        docs_dict[start_year].append(doc)
    return [docs_dict, [len(doc_list) for year, doc_list in docs_dict.items()]]
