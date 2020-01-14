#!/usr/bin/env python3

import gensim
import sys, os
from utils import *

def dump_w2v(model_paths=None, model_dict=None):
    print(timestamp(), "Starting txt and tsv file dump...", file=sys.stderr)

    if not (model_paths or model_dict):
        print("dump_w2v: input list of model paths or valid models.", file=sys.stderr)
    loading = False if model_dict else True

    iter_list = model_dict if model_dict else model_paths

    for obj in iter_list:
        model_path = obj if loading else model_dict[obj]["model_path"]

        base = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        model_txt_path = os.path.join(base, model_name.split(".model")[0] + ".txt")

        # Dump model binary to text file of word2vec format
        model = gensim.models.Word2Vec.load(model_path) if loading else model_dict[obj]["model"]
        vectors = model.wv
        if loading:
            print(timestamp(), "Model loaded from " + model_path, file=sys.stderr)
        vectors.save_word2vec_format(model_txt_path, binary=False)
        print(timestamp(), "Model text file saved to " + model_txt_path, file=sys.stderr)


        # Convert text file to tsv file
        model_tsv_path = os.path.join(base, model_name.split(".model")[0] + ".tsv")
        labels_tsv_path = os.path.join(base, model_name.split(".model")[0] + "_labels.tsv")
        model_tsv_output = []
        labels_tsv_output = []
        first = True
        with open(model_txt_path) as f:
            for line in f:
                if first:
                    first = False
                    continue
                line = line.strip().split()
                model_tsv_output.append("\t".join(line[1:]))
                labels_tsv_output.append(line[0])

        # Write to separate tsv files, readable by projector.tensorflow
        with open(model_tsv_path, "w") as f:
            f.write("\n".join(model_tsv_output))
        print(timestamp(), "Model tsv file saved to " + model_tsv_path, file=sys.stderr)

        with open(labels_tsv_path, "w") as f:
            f.write("\n".join(labels_tsv_output))
        print(timestamp(), "Labels tsv file saved to " + labels_tsv_path, file=sys.stderr)
        print(timestamp(), "Done!", file=sys.stderr)

def main():
    dump_w2v(sys.argv[1:])

if __name__ == '__main__':
    main()
