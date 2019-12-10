#!/usr/bin/env python3
import os, argparse, natsort
from datetime import datetime

def timestamp():
	return "["+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"]"

def get_order(file):
    base = os.path.basename(file)
    if base[:2] == "OA":
        return base[2:]
    return base

def get_year(file):
    """
        Get the year a file refers to.

        file: input filepath of format "OAYYYYMMDD.txt" or "YYYYMMDD.txt"

        returns year in int format
    """
    offset = 2 if os.path.basename(file)[:2] == "OA" else 0
    return int(os.path.basename(file)[0 + offset:4 + offset])

def order_files(args):
    """
        Sort a list of input files by years.

        input: args
        output: dictionary of format {"YYYY":[file0,file1,...], "YYYY+args.year_split":[file0,file1,...]}
    """
    files = [os.path.join(args.corpus_dir, f) for f in os.listdir(args.corpus_dir)
             if (os.path.isfile(os.path.join(args.corpus_dir, f)) and f.endswith('.txt'))]

    files = natsort.natsorted(files, key=lambda x: get_order(x))  # Sort in ascending numeric order
    start_year = get_year(files[0])
    files_dict = {start_year:[]}
    for file in files:
        cur_year = get_year(file)
        if cur_year - start_year >= args.year_split:
            start_year = cur_year
            files_dict[start_year] = []
        files_dict[start_year].append(file)
    return files_dict
