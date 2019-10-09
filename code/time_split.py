#!/usr/bin/env python3

###############################################################################
# time_split.py
#
# Split given directory into chunks based on time. (For Derek Greene's method)
#
###############################################################################
import argparse, numpy, os

def split_into_chunks(args, files):
    """
        Split files into args.num_chunks roughly even chunks of files.

        args: command line arguments
        files: list of files
    """
    files_per_chunk = args.num_chunks // len(files)
    print("Splitting into", args.num_chunks, "chunks of time...")

    # Split files into even sized lists
    chunk_list = numpy.array_split(files, args.num_chunks)
    chunk_list = [chunk.tolist() for chunk in chunk_list]

    # Iterate and create num_chunks directories to put split data into
    for idx in range(args.num_chunks):
        chunk_path = args.corpus_dir + "-" + str(idx)
        if not os.path.exists(chunk_path):
            os.mkdir(chunk_path)
        else:
            print("Directory exists. Overwriting files.")

        # Iterate over files in this given chunk of time
        for file in chunk_list[idx]:
            from_path = os.path.join(args.corpus_dir, file)
            to_path = os.path.join(chunk_path, file)
            os.popen("cp " + from_path + " " + os.path.join(chunk_path, file))


def split_into_years(args, files):
    """
        Split files into directories based on date of file. Will split into
        directories containing files with range args.years_per_chunk.

        args: command line arguments
        files: list of files
    """
    year_idx = 0
    first_year = int(files[0][0:4]) # Find first year of earliest file
    chunk_path = args.corpus_dir + "-" + str(first_year)
    if not os.path.exists(chunk_path):
        os.mkdir(chunk_path)
    else:
        print("Directory exists. Overwriting files.")

    for file in files:
        year_idx = int(file[0:4]) - first_year

        # If we've surpassed the time frame, move onto new directory
        if year_idx >= args.years_per_chunk:
            first_year = int(file[0:4])

            # Define new path to copy files into
            chunk_path = args.corpus_dir + "-" + str(first_year)
            if not os.path.exists(chunk_path):
                os.mkdir(chunk_path)
            else:
                print("Directory exists. Overwriting files.")

        # Copy file into new directory
        from_path = os.path.join(args.corpus_dir, file)
        to_path = os.path.join(chunk_path, file)
        os.popen("cp " + from_path + " " + os.path.join(chunk_path, file))


def main(args):
    # Make list of all files in input directory
    files = [f for f in os.listdir(args.corpus_dir)
             if os.path.isfile(os.path.join(args.corpus_dir, f))]

    # Split based on input arguments
    if args.num_chunks:
        split_into_chunks(args, files)
    else:
        split_into_years(args, files)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, default="../data/sessionsPapers-txt-tok", help='directory containing corpus')
    parser.add_argument('--num_chunks', type=int, default=0, help='number of chunks to split data into')
    parser.add_argument('--years_per_chunk', type=int, default=50, help='number of years per split')
    args = parser.parse_args()
    main(args)
