#!/usr/bin/env python3

import gensim
import sys, os

def dump_w2v(list_of_model_paths):
    for model_path in list_of_model_paths:
        base = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        model_txt_path = os.path.join(base, model_name + ".txt")

        # Dump model binary to text file of word2vec format
        model = gensim.models.Word2Vec.load(model_path)
        vectors = model.wv
        print("Model loaded from " + model_path, file=sys.stderr)
        vectors.save_word2vec_format(model_txt_path, binary=False)
        print("Model text file saved to " + model_txt_path, file=sys.stderr)


        # Convert text file to tsv file
        model_tsv_path = os.path.join(base, model_name + ".tsv")
        labels_tsv_path = os.path.join(base, model_name + "_labels.tsv")
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
        print("Model tsv file saved to " + model_tsv_path, file=sys.stderr)

        with open(labels_tsv_path, "w") as f:
            f.write("\n".join(labels_tsv_output))
        print("Labels tsv file saved to " + labels_tsv_path, file=sys.stderr)
        print("Done!", file=sys.stderr)

def main():
    dump_w2v(sys.argv[1:])

if __name__ == '__main__':
    main()
