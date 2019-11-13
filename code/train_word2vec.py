#!/usr/bin/env python3

import os, sys, argparse, time
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def build_corpus(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                 if (os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt'))]
    corpus = []
    for file in files:
        with open(file) as f:
            corpus.append(f.read().split())
    return corpus

# https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/notebook
def tsne_plot(model, pre):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()
    plt.savefig(pre + 'plot.png')

def main(args):
    print("Building corpus...", file=sys.stderr)
    corpus = build_corpus(args.corpus_dir)

    model = Word2Vec(min_count=1)#, size=100, window=20)#, workers=4)
    print("Building vocab...", file=sys.stderr)
    model.build_vocab(corpus)
    print("Training model...", file=sys.stderr)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    pre = args.save_model_dir + "word2vec/" + time.strftime("%Y-%m-%d") + "/" + time.strftime("%H-%M-%S") + "/"
    if not os.path.exists(pre):
        os.makedirs(pre)

    print("Visualizing results...", file=sys.stderr)
    tsne_plot(model, pre)

    model.save(pre + "model")
    print("Results saved to directory " + pre, file=sys.stderr)
    print("Done!", file=sys.stderr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt-tok-dh", help='directory containing corpus')
    parser.add_argument('--save_model_dir', type=str, default="../models/", help='base directory for saving model directory')
    args = parser.parse_args()
    main(args)
