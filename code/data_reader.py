#!/usr/bin/env python3

import sys
import os
from tqdm import tqdm
from tei_reader import TeiReader

def main():
    # Load data directory and pick output file
    try:
        XML_data_dir = sys.argv[1]
    except:
        print("Input directory of XML files")
        sys.exit(1)

    input_files = [os.path.join(XML_data_dir, f) for f in os.listdir(XML_data_dir) if os.path.isfile(os.path.join(XML_data_dir, f))]
    txt_output_dir = os.path.dirname(XML_data_dir) + "-txt"
    if not os.path.exists(txt_output_dir):
        os.mkdir(txt_output_dir)

    # Initialize reader for data
    reader = TeiReader()

    # Go through input files and generate output files
    for i in tqdm(range(len(input_files))):
        # Get current file
        file = input_files[i]

        # Change to txt file
        filename = os.path.splitext(os.path.basename(file))[0] + ".txt"
        file_path = os.path.join(txt_output_dir, filename)
        if os.path.exists(file_path):
            continue

        # Write text to txt file
        with open(os.path.join(txt_output_dir, filename), "w+") as f:
            # Get text from tei_reader
            try:
                text = reader.read_file(file)
            except UnicodeDecodeError:
                print("Error reading " + file + ". Skipping...")
                continue
            f.write(text.text)


if __name__ == '__main__':
    main()
