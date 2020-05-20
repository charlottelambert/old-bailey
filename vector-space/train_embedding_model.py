#!/usr/bin/env python3

import os, sys, argparse, time, csv
from gensim.models import Word2Vec, FastText
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from dump_w2v import dump_w2v
from wordcloud import WordCloud
import pandas as pd
from gensim.models import KeyedVectors
sys.path.append('../')
from utils import *

word_list = ["sentence", "punishment", "guilt", "murder", "vote", "woman", "man", "innocent", "london", "crime", "female", "slave", "chattle", "foreigner", "foreign",  "theft", "robbery", "rape", "thievery", "larceny", "burglary", "assault", "hanging", "prison", "convict"]
word_list.sort()

def gen_wordcloud(args, pre, first_year, neighbor_dict):
    # dict is {word: [(neighbor, similarity), ...]}
    # pre will be /path/to/dir/
    max_words = len(word_list)*(args.find_n_neighbors+1)
    wordcloud = WordCloud(background_color="white", max_words=max_words, width=400, height=400)

    word_frequency_list = []
    for word in neighbor_dict:
        word_frequency_list.append((word, 1))
        word_frequency_list += neighbor_dict[word]

    wordcloud = wordcloud.fit_words(dict(word_frequency_list))
    wc_path = os.path.join(pre, word + "_" + str(first_year) + ".jpg")
    wordcloud.to_file(wc_path)
    print(timestamp(), "Saving word cloud to", wc_path, file=sys.stderr)


def find_n_neighbors(args, pre, model_dict):
    print(timestamp(), "Finding nearest", args.find_n_neighbors, "neighbors...", file=sys.stderr)
    tsv_path = os.path.join(pre, "neighbors.tsv")
    with open(tsv_path, "w") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        label_row = ["word"] + [first_year for first_year, model_info in model_dict.items()]
        tsv_writer.writerow(label_row)
        neighbor_dict = {}

        for word in word_list:
            row_contents = [word]
            for first_year, model_info in model_dict.items():
                try:
                    neighbor_list = model_info["model"].similar_by_word(word, args.find_n_neighbors)
                    row_contents.append(neighbor_list)
                    neighbor_dict[first_year] = {}
                    neighbor_dict[first_year][word] = neighbor_list
                    gen_wordcloud(args, pre, first_year, neighbor_dict[first_year])

                except KeyError:
                    print("Word \"" + word + "\" not in vocabulary. Skipping...", file=sys.stderr)
                    row_contents.append("UNK")
            tsv_writer.writerow(row_contents)

    print(timestamp() + " Wrote top", args.find_n_neighbors, "neighbors to", tsv_path, file=sys.stderr)
    return neighbor_dict

def build_corpus(args, input_dir_path=None, files=None, corpus_txt_file=None):
    """
        Function to build a corpus (list of contents of files) based on input
        directory.
    """
    # If input tsv file, compile the third column of all
    if args.tsv_corpus:
        corpus = []
        for line in files:
            text = line.split("\t")[2:]
            corpus.append("\t".join(text).lower().split())
    elif not files and input_dir_path:
        files = [os.path.join(input_dir_path, f) for f in os.listdir(input_dir_path)
                     if (os.path.isfile(os.path.join(input_dir_path, f)) and f.endswith('.txt'))]
        corpus = []
        for file in files:
            with open(file) as f:
                corpus.append(f.read().lower().split())
    elif files:
        corpus = []
        for file in files:
            with open(file) as f:
                corpus.append(f.read().lower().split())
    # If input corpus.txt file output from Mallet's --print-output flag
    elif corpus_txt_file:
        print(timestamp(),"Building corpus from file...", file=sys.stderr)
        corpus = []
        doc_words = []
        with open(corpus_txt_file, 'r') as f:
            content = f.read()
            lines = content.split("\n")
            for line in lines:
                if re.match("[0-9]+:", line):
                    word = line.split(" ")[1]
                    doc_words.append(word)
                elif line[:5] == "name:" and len(doc_words) > 0:
                    corpus.append(doc_words)
                    doc_words = []
                else: continue
    else:
        print("build_corpus(): Please input a path to a directory, list of files, or corpus file.", file=sys.stderr)
        exit(1)
    return corpus

# https://stackoverflow.com/questions/48941648/how-to-remove-a-word-completely-from-a-word2vec-model-in-gensim
def filter_top_words(model, n):
    """
        Filter the most frequent n words found by model. Maximum of 10,000 words
        can be visualized using projector.tensorflow.org.
    """
    wv = model.wv
    words_to_trim = wv.index2word[n:]
    ids_to_trim = [wv.vocab[w].index for w in words_to_trim]

    for w in words_to_trim:
        del wv.vocab[w]

    wv.vectors = np.delete(wv.vectors, ids_to_trim, axis=0)
    wv.init_sims(replace=True)

    for i in sorted(ids_to_trim, reverse=True):
        del(wv.index2word[i])
    return model

# https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/notebook
def tsne_plot(model, pre, neighbor_dict):
    """
        Creates and TSNE model and plots it
    """
    words_to_plot = []
    for word in neighbor_dict:
        words_to_plot.append(word)
        for neighbor in neighbor_dict[word]:
            words_to_plot.append(neighbor[0])

    labels = []
    tokens = []

    for i, word in enumerate(model.wv.vocab):
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(random_state=2017, perplexity=12, n_components=2, init='pca', method='barnes_hut', verbose=1)
    print(timestamp(), "TSNE model initialized.", file=sys.stderr)
    try:
        new_values = tsne_model.fit_transform(tokens)
    except KeyboardInterrupt:
        print(timestamp(), "Exiting...", file=sys.stderr)

    x = []
    y = []
    for i in tqdm(range(len(new_values))):
        value = new_values[i]
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))
    for i in tqdm(range(len(x))):
        if labels[i] not in words_to_plot:
            continue

        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(pre + "_plot.png")
    print(timestamp(), "TSNE plot saved.", file=sys.stderr)

def main(args):
    print(timestamp(), "Beginning at " + time.strftime("%m/%d/%Y %H:%M "), file=sys.stderr)
    model_base = "fasttext/" if args.f else "word2vec/"
    embedding_model = FastText if args.f else Word2Vec

    if args.load_model_dir:
        models = [os.path.join(args.load_model_dir, f) for f in os.listdir(args.load_model_dir)
                 if os.path.join(args.load_model_dir, f)[-6:] == ".model"]
        model_dict = {}
        for model_path in models:
            model_name = os.path.basename(model_path).split(".model")[0]
            model = embedding_model.load(model_path)
            model_dict[model_name] = {"model": model, "model_path": model_path}
            print(timestamp(), "Model loaded from " + model_path, file=sys.stderr)
            pre = os.path.dirname(model_path)
            if args.print_similarity:
                try:
                    print("Similarity between \'murder\' and \'murther\':",model.similarity('murder', 'murther'))
                except KeyError as e:
                    print("ERROR:", e)
        if args.print_similarity: exit(0)
    else:
        pre = args.save_model_dir + model_base + time.strftime("%Y-%m-%d") + "/" + time.strftime("%H-%M-%S") + "/"
        if not os.path.exists(pre):
            os.makedirs(pre)
        print(timestamp(), "Writing all files to", pre, file=sys.stderr)

        if args.corpus_txt_file:
            corpus = build_corpus(args,corpus_txt_file=args.corpus_txt_file)
            model = embedding_model(min_count=1)#, size=100, window=20)#, workers=4)
            print(timestamp(), "Building vocab...", file=sys.stderr)
            model.build_vocab(corpus)
            msg = "Training model..."
            # Filter out top words (need to filter to 10000 if using projector.tensorflow)
            if args.filter_top_words:
                print(timestamp(), "Extracting top " + str(args.filter_top_words) + " words...", file=sys.stderr)
                model = filter_top_words(model, args.filter_top_words)

            print(timestamp(), msg, file=sys.stderr)
            model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs)
            try:
                print("Similarity between \'murder\' and \'murther\':",model.similarity('murder', 'murther'))
            except KeyError as e:
                print("ERROR:", e)
            model_path = os.path.join(pre, "w2v.model")
            model.save(model_path)
            print(timestamp(), "Model saved to", model_path, file=sys.stderr)
            dump_w2v(model_paths=[model_path])
            exit(0)

        else:
            # Order files by year
            files_dict, _ = order_files(args)

            print(timestamp(), "Data will be saved to directory " + pre, file=sys.stderr)
            model_dict = {}
            for first_year, file_list in files_dict.items():
                print(timestamp(), "Building corpus...", file=sys.stderr)
                corpus = build_corpus(args, files=file_list)
                if args.pretrained:
                    print(timestamp(), "Initializing model with corpus...", file=sys.stderr)
                    model = embedding_model(corpus, size=300) #, min_count=1)#, size=100, window=20)#, workers=4)
                    print(timestamp(), "Intersecting with pretrained model", file=sys.stderr)
                    model.intersect_word2vec_format(args.pretrained,
                                        lockf=1.0,
                                        binary=True)
                    msg = "Retraining model..."
                else:
                    model = embedding_model(min_count=1)#, size=100, window=20)#, workers=4)
                    print(timestamp(), "Building vocab...", file=sys.stderr)
                    model.build_vocab(corpus)

                    msg = "Training model..."
                # Filter out top words (need to filter to 10000 if using projector.tensorflow)
                if args.filter_top_words:
                    print(timestamp(), "Extracting top " + str(args.filter_top_words) + " words...", file=sys.stderr)
                    model = filter_top_words(model, args.filter_top_words)

                print(timestamp(), msg, file=sys.stderr)
                model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs)
                try:
                    print("Similarity between \'murder\' and \'murther\':",model.similarity('murder', 'murther'))
                except KeyError as e:
                    print("ERROR:", e)
                model_path = os.path.join(pre, str(first_year)) + ".model"
                model.save(model_path)
                model_dict[first_year] = {"model": model, "model_path": model_path}
                print(timestamp(), "Model saved to", model_path, file=sys.stderr)

            dump_w2v(model_dict=model_dict)
            print("model dict:", model_dict)

    # Find nearest n neighbors
    if args.find_n_neighbors or args.plot_neighbors:
        neighbor_dict = find_n_neighbors(args, pre, model_dict)

    # THIS IS BROKEN
    if args.plot_neighbors:
        print(timestamp(), "Visualizing results...", file=sys.stderr)
        for year in model_dict:
            tsne_plot(model_dict[year]["model"], os.path.join(pre, str(year)), neighbor_dict[year])

    print(timestamp(), "Done! Ending at " + time.strftime("%d/%m/%Y %H:%M ") , file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_corpus', type=str, default="", help='path to corpus file in tsv format')
    parser.add_argument('--corpus_txt_file', type=str, default="", help='path to corpus file saved from mallet --print_output')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-lower-lemma", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--load_model_dir', type=str, help='path to directory containing models to load and visualize.')
    parser.add_argument('--pretrained', type=str, help='path to pretrained model (use /work/clambert/models/pretrained/GoogleNews-vectors-negative300.bin).')
    parser.add_argument('--plot_neighbors', default=False, action="store_true", help='whether or not to visualize and plot data.')
    parser.add_argument('--filter_top_words', type=int, default=10000, help='number of words to include in model (take the most common words)')
    parser.add_argument('--find_n_neighbors', type=int, default=0, help='how many nearest neighbors to find')
    parser.add_argument('--epochs', type=int, default=100, help='how many epochs')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to include in each chunk of corpus (run tf-idf over each chunk)')
    parser.add_argument('-f', action='store_true', help='use fasttext model instead of word2vec')
    parser.add_argument('--print_similarity', action='store_true', default=False, help='whether or not to print out similarities')
    args = parser.parse_args()
    main(args)
