#!/usr/bin/env python3
import gensim
import sys, os, argparse
sys.path.append('../')
from utils import *

def detm_embed_dump(model_txt_path):
    """
        Dump appropriate files to use projector.tensorflow.org for
        visualization on embeddings from detm.

        input:
            model_txt_path (str): path to embeddings file
    """
    # Convert text file to tsv file
    model_tsv_path = model_txt_path.rstrip("/") + ".tsv"
    labels_tsv_path = model_txt_path.rstrip('/') + "_labels.tsv"
    model_tsv_output = []
    labels_tsv_output = []
    with open(model_txt_path) as f:
        for line in f:
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

def dump_w2v(model_paths=None, model_dict=None, detm=False):
    """
        Dump appropriate files to use projector.tensorflow.org for
        visualization on model trained by train_embeddings.py.

        input:
            model_paths (list): optional list of models to load and dump
            model_dict (dict): optional dictionary of models to load and write
                files for
            detm (bool): optional flag indicating whether or not embeddings
                were trained by detm code
    """
    print(timestamp(), "Starting txt and tsv file dump...", file=sys.stderr)
    if not (model_paths or model_dict):
        print("dump_w2v: input list of model paths or valid models.", file=sys.stderr)
    loading = False if model_dict else True

    iter_list = model_dict if model_dict else model_paths

    for obj in iter_list:
        model_path = obj if loading else model_dict[obj]["model_path"]
        if detm:
            detm_embed_dump(model_path)
            continue

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
    dump_w2v([args.model_path], detm=args.detm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, default="", help='path to word2vec model/embedding model run using detm')
    parser.add_argument('--detm', action='store_true', help='flag indicating that input model was run for detm')
    args = parser.parse_args()
    main(args)
