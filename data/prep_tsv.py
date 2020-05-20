#!/usr/bin/env python3
import sys, optparse, os,shutil
import numpy as np
sys.path.append("..")
from utils import timestamp

def main(options, args):
    if len(args) > 0:
        print(timestamp(), "Merging TSV files...", file=sys.stderr)
        base = os.path.dirname(args[0])
        suffix = os.path.basename(args[0]).split("-")[2:]

        output_file = base.replace(".tsv-dir", "-" + "-".join(suffix))
        print(timestamp(), "Merged tokenized TSV file being written to", output_file)
        all_lines = []
        for file in args:
            if "split-" not in os.path.basename(file): continue
            with open(file, 'r') as f:
                all_lines += f.read().split("\n")
        # all_lines = [header] + all_lines
        with open(output_file,'w') as f:
            f.write("\n".join(all_lines))
        if options.rm_dir: shutil.rmtree(base)
        print(timestamp(), "Done!", file=sys.stderr)
    elif options.tsv_corpus:
        base = os.path.basename(options.tsv_corpus) + "-dir"
        new_dir = os.path.join(os.path.dirname(options.tsv_corpus), base)
        if not os.path.exists(new_dir): os.makedirs(new_dir)
        else:
            print("Directory", new_dir, "already exists. Exiting...", file=sys.stderr)
            exit(0)

        print(timestamp(), "Splitting TSV file...", file=sys.stderr)
        with open(options.tsv_corpus, 'r') as f:
            lines = f.read().split("\n")
            if lines[0].lower() == "id\tyear\ttext": idx = 1
            else: idx = 0
            n = len(lines[idx:]) // options.num_splits
            splits = [lines[idx:][i * n:(i + 1) * n]
                     for i in range((len(lines[idx:]) + n - 1) // n)]

            # Make sure no data was lost in the splits
            try:
                assert sum([len(l) for l in splits]) == len(lines[idx:])
            except AssertionError:
                print(timestamp(), "Lost some lines, not sure why. Exiting...", file=sys.stderr)

            for i, s in enumerate(splits):
                # Unique file path
                out_file = os.path.join(new_dir, "split-" + str(i) + ".tsv")
                with open(out_file, 'w') as f:
                    f.write("\n".join(s))
        print(timestamp(), "Done!")
    else:
        print("Input files to merge or file to split.", file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="usage: %prog [options] tsv_corpus1 tsv_corpus2 ...")
    parser.add_option('--rm_dir', default=False, action='store_true', help='whether or not to remove directory after merging')
    parser.add_option('--tsv_corpus', type=str, default='', help='path to tsv file to split')
    parser.add_option('--num_splits', type=int, default=4, help='how many tsv files to split options.tsv_corpus into')
    (options, args) = parser.parse_args()
    main(options, args)
