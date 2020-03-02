#!/usr/bin/env python3
import os, argparse, natsort
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
    return "["+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"]"

def get_order(file):
    base = os.path.basename(file)
    if base[:2] == "OA":
        return base[2:]
    return base

def get_year(file, include_month=False):
    """
        Get the year a file refers to.

        file: input filepath of format "OAYYYYMMDD.txt" or "YYYYMMDD.txt"

        returns year in int format
    """
    offset = 2 if os.path.basename(file)[:2] == "OA" else 0
    year = int(os.path.basename(file)[0 + offset:4 + offset])
    if include_month:
        month = int(os.path.basename(file)[4 + offset:6 + offset])
        return (year, month)
    return year

def order_files(args, ret_dict=True):
    """
        Sort a list of input files by years.

        input: args
        output: dictionary of format {"YYYY":[file0,file1,...], "YYYY+args.year_split":[file0,file1,...]}
    """
    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f))
                 and re.match(".*[0-9]{8}", f) and f.endswith('.txt'))]

    files = natsort.natsorted(files, key=lambda x: get_order(x))  # Sort in ascending numeric order
    start_year = get_year(files[0])
    files_dict = {start_year:[]}
    for file in files:
        cur_year = get_year(file)
        if cur_year - start_year >= args.year_split:
            start_year = cur_year
            files_dict[start_year] = []
        files_dict[start_year].append(file)
    if ret_dict:
        return files_dict
    else:
        # [ordered_file_list, [docs_per_slice]]
        return [files, [len(files_dict[chunk]) for chunk in files_dict]]
