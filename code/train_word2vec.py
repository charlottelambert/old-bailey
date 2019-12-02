#!/usr/bin/env python3

import os, sys, argparse, time
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def build_corpus(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                 if (os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt'))]
    corpus = []
    for file in files:
        with open(file) as f:
            corpus.append(f.read().split())
    return corpus

# https://stackoverflow.com/questions/48941648/how-to-remove-a-word-completely-from-a-word2vec-model-in-gensim
def filter_top_words(model, n):
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
def tsne_plot(model, pre):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    for i, word in enumerate(model.wv.vocab):
        tokens.append(model.wv[word])
        labels.append(word)

#    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2500, random_state=23)
    tsne_model = TSNE(random_state=2017, perplexity=12, n_components=2, init='pca', method='barnes_hut', verbose=1)
    print("TSNE model initialized.", file=sys.stderr)
    # PROBLEM with fit_transform, it's just never returning
    try:
        new_values = tsne_model.fit_transform(tokens)
    except KeyboardInterrupt:
        print("Exiting...", file=sys.stderr)

    x = []
    y = []
#    for value in new_values:
    for i in tqdm(range(len(new_values))):
        value = new_values[i]
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))
    for i in tqdm(range(len(x))):
    #for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()
    plt.savefig(os.path.join(pre, 'plot.png'))
    print("TSNE plot saved.", file=sys.stderr)

def main(args):
    print("Beginning at " + time.strftime("%d/%m/%Y %H:%M "), file=sys.stderr)
    if not  args.load_model:
        pre = args.save_model_dir + "word2vec/" + time.strftime("%Y-%m-%d") + "/" + time.strftime("%H-%M-%S") + "/"
        if not os.path.exists(pre):
            os.makedirs(pre)
        print("Data will be saved to directory " + pre, file=sys.stderr)
        print("Building corpus...", file=sys.stderr)
        corpus = build_corpus(args.corpus_dir)

        model = Word2Vec(min_count=1)#, size=100, window=20)#, workers=4)
        print("Building vocab...", file=sys.stderr)
        model.build_vocab(corpus)

        # Filter out top words (need to filter to 10000 if using projector.tensorflow)
        if args.filter_top_words:
            print("Extracting top " + str(args.filter_top_words) + " words...", file=sys.stderr)
            model = filter_top_words(model, args.filter_top_words)
        print("Training model...", file=sys.stderr)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

        model.save(pre + "model")
        print("Model saved. ", file=sys.stderr)
    else:
        # Load model from args.load_model
        model = Word2Vec.load(args.load_model)
        print("Model loaded from " + args.load_model, file=sys.stderr)
        pre = os.path.dirname(args.load_model)

    if args.plot:
        print("Visualizing results...", file=sys.stderr)
        tsne_plot(model, pre)

    print("Done! Ending at " + time.strftime("%d/%m/%Y %H:%M ") , file=sys.stderr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsAndOrdinarys-txt-tok-dh", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--load_model', type=str, help='path to model to load and visualize.')
    parser.add_argument('--plot', default=False, action="store_true", help='whether or not to visualize and plot data.')
    parser.add_argument('--filter_top_words', type=int, default=10000, help='number of words to include in model (take the most common words)')
    args = parser.parse_args()
    main(args)
