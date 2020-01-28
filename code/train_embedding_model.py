#!/usr/bin/env python3

import os, sys, argparse, time, csv
from gensim.models import Word2Vec, FastText
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from utils import *
from dump_w2v import dump_w2v
from wordcloud import WordCloud
import pandas as pd

word_list = ["sentence", "punishment", "guilt", "murder", "vote", "woman", "man", "innocent", "London", "crime", "female", "slave", "chattle", "foreigner", "foreign",  "theft", "robbery", "rape", "thievery", "larceny", "burglary", "assault", "hanging", "prison", "convict"]
word_list.sort()

def gen_wordcloud(args, pre, neighbor_dict):
    # dict is {word: [(neighbor, similarity), ...]}
    # pre will be /path/to/dir/1674
    max_words = len(word_list)*(args.find_n_neighbors+1)
    # wc = WordCloud(background_color="white", max_words=max_words, width=400, height=400, random_state=1).generate(text)
    # # to recolour the image
    # # plt.imshow(wc.recolor(color_func=image_colors))
    # plt.imshow()

    wordcloud = WordCloud(background_color="white", max_words=max_words, width=400, height=400)

    word_frequency_list = []
    for word in neighbor_dict:
        word_frequency_list.append((word, 1))
        word_frequency_list += neighbor_dict[word]

    wordcloud = wordcloud.fit_words(dict(word_frequency_list))
    wc_path = pre + "_" + word + "_wordcloud.jpg"
    wordcloud.to_file(wc_path)
    print(timestamp(), "Saving word cloud to", wc_path, file=sys.stderr)


# want to do this once for all models trained
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
                    gen_wordcloud(args, os.path.join(pre, str(first_year)), neighbor_dict[first_year])

                except KeyError:
                    print("Word \"" + word + "\" not in vocabulary. Skipping...", file=sys.stderr)
                    row_contents.append("UNK")
            tsv_writer.writerow(row_contents)

    print(timestamp() + " Wrote top", args.find_n_neighbors, "neighbors to", tsv_path, file=sys.stderr)
    return neighbor_dict

def build_corpus(input_dir_path=None, files=None):
    """
        Function to build a corpus (list of contents of files) based on input
        directory.
    """
    if not files and input_dir_path:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if (os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt'))]
    elif not input_dir_path and not files:
        print("build_corpus(): Please input a path to a directory or a list of files.", file=sys.stderr)
        exit(1)
    corpus = []
    for file in files:
        with open(file) as f:
            corpus.append(f.read().split())
    return corpus

# want to do this for each model trained
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

# want to do this for each model trained
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

    # # problem: this is plotting all the stuff by year
    # for first_year in neighbor_dict:
    #     # Reset labels
    labels = []
    tokens = []

        # for word in neighbor_dict[first_year]:
        #     try:
        #         tokens.append(model.wv[word])
        #         labels.append(word)
        #     except KeyError:
        #         continue
        #     for neighbor in neighbor_dict[first_year][word]:
        #         tokens.append(neighbor[1])
        #         labels.append(neighbor[0])


        # ###############
    for i, word in enumerate(model.wv.vocab):
        # print(word)
        # exit(0)
        tokens.append(model.wv[word])
        labels.append(word)

#   tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2500, random_state=23)
    tsne_model = TSNE(random_state=2017, perplexity=12, n_components=2, init='pca', method='barnes_hut', verbose=1)
    print(timestamp(), "TSNE model initialized.", file=sys.stderr)
    # PROBLEM with fit_transform, it's just never returning
    try:
        new_values = tsne_model.fit_transform(tokens)
    except KeyboardInterrupt:
        print(timestamp(), "Exiting...", file=sys.stderr)

    x = []
    y = []
#    for value in new_values:
    for i in tqdm(range(len(new_values))):
        value = new_values[i]
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))
    for i in tqdm(range(len(x))):
        if labels[i] not in words_to_plot:
            continue
    #for i in range(len(x)):

        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()
    plt.savefig(pre + "_plot.png")
    print(timestamp(), "TSNE plot saved.", file=sys.stderr)

def main(args):
    print(timestamp(), "Beginning at " + time.strftime("%m/%d/%Y %H:%M "), file=sys.stderr)
    model_base = "fasttext/" if args.f else "word2vec/"
    embedding_model = FastText if args.f else Word2Vec
    print("Embedding model:", embedding_model)

    if not args.load_model_dir:
        pre = args.save_model_dir + model_base + time.strftime("%Y-%m-%d") + "/" + time.strftime("%H-%M-%S") + "/"
        if not os.path.exists(pre):
            os.makedirs(pre)

        # Order files by year
        files_dict = order_files(args)

        print(timestamp(), "Data will be saved to directory " + pre, file=sys.stderr)
        model_dict = {}
        for first_year, file_list in files_dict.items():
            print(timestamp(), "Building corpus...", file=sys.stderr)
            corpus = build_corpus(files=file_list)

            model = embedding_model(min_count=1)#, size=100, window=20)#, workers=4)
            print(timestamp(), "Building vocab...", file=sys.stderr)
            model.build_vocab(corpus)

            # Filter out top words (need to filter to 10000 if using projector.tensorflow)
            if args.filter_top_words: # Should this only happen with word2vec and not fasttext?
                print(timestamp(), "Extracting top " + str(args.filter_top_words) + " words...", file=sys.stderr)
                model = filter_top_words(model, args.filter_top_words)

            print(timestamp(), "Training model...", file=sys.stderr)
            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model_path = os.path.join(pre, str(first_year)) + ".model"
            model.save(model_path)
            model_dict[first_year] = {"model": model, "model_path": model_path}
            print(timestamp(), "Model saved to", model_path, file=sys.stderr)
        dump_w2v(model_dict=model_dict)
        print("model dict:", model_dict)

    else: # FIX THIS SO IT CAN CALL FIND N NEIGHBORS PROPERLY
        # Load model from args.load_model_dir MAKE THIS LAOD A MODEL DIR INSTEAD OF A MODEL!
        models = [os.path.join(args.load_model_dir, f) for f in os.listdir(args.load_model_dir)
                 if os.path.join(args.load_model_dir, f)[-6:] == ".model"]
        model_dict = {}
        for model_path in models:
            model_name = os.path.basename(model_path).split(".model")[0]
            model = embedding_model.load(model_path)
            model_dict[model_name] = {"model": model, "model_path": model_path}
            print(timestamp(), "Model loaded from " + model_path, file=sys.stderr)
            pre = os.path.dirname(model_path)

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
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-dh", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--load_model_dir', type=str, help='path to directory containing models to load and visualize.')
    parser.add_argument('--plot_neighbors', default=False, action="store_true", help='whether or not to visualize and plot data.')
    parser.add_argument('--filter_top_words', type=int, default=10000, help='number of words to include in model (take the most common words)')
    parser.add_argument('--find_n_neighbors', type=int, default=0, help='how many nearest neighbors to find')
    parser.add_argument('--year_split', type=int, default=100, help='number of years to include in each chunk of corpus (run tf-idf over each chunk)')
    parser.add_argument('-f', action='store_true', help='use fasttext model instead of word2vec')
    args = parser.parse_args()
    main(args)
