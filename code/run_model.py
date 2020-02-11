#!/usr/bin/env python3

###############################################################################
# run_model.py
#
# Run given model on input directory. Either use Mallet LDA, gensim's multicore,
# or DTM based on input.
#
###############################################################################

import sys, os, click, csv, gensim, time, argparse
from os import listdir
import custom_stop_words as stop
from gensim import corpora, models
from gensim.test.utils import get_tmpfile
from gensim.corpora import MalletCorpus, Dictionary, bleicorpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers.dtmmodel import DtmModel
from utils import *

# TAKEN OUT OF RUN-LDA.SBATCH, PUT BACK IF RUNNING OUT OF MEMORY
# #SBATCH -c 64
# export JAVA_OPTIONS="-Xms4G -Xmx8G"

# https://markroxor.github.io/gensim/static/notebooks/ldaseqmodel.html

def print_params(pre, args):
    # Print out arguments used to file
    with open(pre + "parameters.tsv", "w+") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        arg_dict = vars(args)
        for key in arg_dict:
            tsv_writer.writerow([key, arg_dict[key]])

def get_ngrams(args, texts):
    if not args.unigrams_only:
        print(timestamp() + " Finding bigrams.", file=sys.stderr)
        bigram = Phrases(texts, min_count=1) # Is this an appropriate value for min_count?
        bigram = Phraser(bigram)
        for idx in range(len(texts)):
            bigrams = bigram[texts[idx]]
            if args.bigrams_only:
                texts[idx] = [] # If we only want bigrams, remove all unigrams
            for token in bigrams:
                if '_' in token:
                    texts[idx].append(token)
                    if args.mixed_ngrams:
                        texts[idx].append(token) # Scale bigrams
    return texts

def model_on_directory(args):
    # Types of valid models to run
    type_list = ["lda", "multicore", "dtm"]
    if args.model_type not in type_list:
        print("Please specify a valid model type (multicore, lda, dtm).", sys.err)
        sys.exit(1)

    # Prefix for running lda (modify if files should go to a different directory)
    # FORMAT: "/work/clambert/models/model_type/YYYYMMDD/HH-MM-SS"
    pre = args.save_model_dir + args.model_type + "/" + time.strftime("%Y%m%d") + "/" + time.strftime("%H-%M-%S") + "/"

    if not os.path.exists(pre):
        os.makedirs(pre)

    print(timestamp() + " Reading corpus.", file=sys.stderr)

#    files = [f for f in os.listdir(args.corpus_dir)
#             if os.path.isfile(os.path.join(args.corpus_dir, f))]
    files, time_slices = order_files(args, ret_dict=False)
    print(timestamp() + " Time slices:", time_slices)
    # Compile list of lists of tokens
    texts = []
    print(timestamp() + " Compiling tokens.", file=sys.stderr)
    for i in range(len(files)):
        file = files[i]
        with open(os.path.join(args.corpus_dir, file)) as f:
            text = f.read().lower().replace("\n", " ").split(" ")

            # Changed: Also remove stop words from Mallet version
            stop_words = stop.modified_stop_words
            text = [word for word in text if word not in stop_words]
            texts.append(text)

    # If we want to include a mix of unigrams and bigrams or just bigrams
    texts = get_ngrams(args, texts)

    print(timestamp() + " Building dictionary.", file=sys.stderr)

    dictionary = corpora.Dictionary(texts)
    if not args.mixed_ngrams:
        # Filter extremes if not doing only bigrams
        dictionary.filter_extremes(no_below=50, no_above=0.90)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # Run the specified model
    if args.model_type == "multicore":
        lda = gensim.models.ldamulticore.LdaMulticore
        ldamodel = lda(corpus, num_topics=args.num_topics,
                       id2word=dictionary, passes=200, alpha=20, workers=8,
                       prefix=pre)
    elif args.model_type == "lda":
        MALLET_PATH = os.environ.get("MALLET_PATH", "~/Mallet/bin/mallet")
        lda = gensim.models.wrappers.LdaMallet
        model = lda(MALLET_PATH, corpus, num_topics=args.num_topics,
                       id2word=dictionary, optimize_interval=args.optimize_interval,
                       workers=12, iterations=args.num_iterations,
                       prefix=pre)
    elif args.model_type == "dtm": # Dynamic Topic Model
        # Find path to DTM binary
        DTM_PATH = os.environ.get('DTM_PATH', None)
        if not DTM_PATH:
            raise ValueError("You need to set the DTM path.")
        # Run the model
        model = DtmModel(DTM_PATH, corpus=corpus,
            id2word=dictionary, time_slices=time_slices, prefix=pre)

    # Save model with timestamp
    model.save(pre + "model")
    print_params(pre, args)

    f = open(pre + "file_ordering.txt", "w+")
    text = ""
    for filename in files:
        text += filename + " "
    f.write(text[:-1])
    f.close()

    print(timestamp() + " Done.", file=sys.stderr)

    return model.print_topics(num_topics=-1, num_words=20)
# _________________________________________________________________________

def main(args):
    arg_dict = vars(args)
    print(timestamp() + " Starting...")
    print(str(arg_dict))
    print(model_on_directory(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--unigrams_only', default=False, action="store_true", help='whether or not to only include unigrams')
    parser.add_argument('--bigrams_only', default=False, action="store_true", help='whether or not to only include bigrams')
    parser.add_argument('--mixed_ngrams', default=False, action="store_true", help='whether or not to include both unigrams and bigrams')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt-tok", help='directory containing corpus')
    parser.add_argument('--model_type', type=str, default="lda", help='type of model to run') # Include dynamic here?
    parser.add_argument('--num_topics', type=int, default=20, help='number of topics to find')
    parser.add_argument('--optimize_interval', type=int, default=10, help='number of topics to find')
    parser.add_argument('--num_iterations', type=int, default=1000, help='number of topics to find')
    parser.add_argument('--year_split', type=int, default=100, help='Number of years per time slice')
    args = parser.parse_args()
    main(args)
