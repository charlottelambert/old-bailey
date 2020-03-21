#!/usr/bin/env python3

###############################################################################
# run_model.py
#
# Run given model on input directory. Either use Mallet LDA, gensim's multicore,
# or DTM based on input.
#
###############################################################################

import sys, os, click, csv, gensim, time, argparse, pyLDAvis, tempfile, random
# import custom_stop_words as stop
from gensim import corpora, models
from gensim.test.utils import get_tmpfile
from gensim.corpora import MalletCorpus, Dictionary, bleicorpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.models.ldaseqmodel import LdaSeqModel
from utils import *
from mallet import Mallet
from techknacq.corpus import Corpus

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

def compile_tokens(args, files):
    # Compile list of lists of tokens
    texts = []
    print(timestamp() + " Compiling tokens.", file=sys.stderr)
    for i in range(len(files)):
        file = files[i]
        with open(os.path.join(args.corpus_dir, file)) as f:
            text = f.read().lower().replace("\n", " ").split(" ")

            # Changed: Also remove stop words from Mallet version
            # stop_words = stop.modified_stop_words
            # text = [word for word in text if word not in stop_words]
            texts.append(text)

    # If we want to include a mix of unigrams and bigrams or just bigrams
    texts = get_ngrams(args, texts)
    return texts

def run_lda(args, corpus, pre, dictionary=None, workers=None):
    MALLET_PATH = os.environ.get("MALLET_PATH", "lda-tools/ext/mallet/bin/mallet")
    if args.gensim:
        lda = gensim.models.wrappers.LdaMallet
        model = lda(MALLET_PATH, corpus, num_topics=args.num_topics,
                       id2word=dictionary, optimize_interval=args.optimize_interval,
                       workers=workers, iterations=args.num_iterations,
                       prefix=pre)
    else:
        rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '-'
        prefix = os.path.join(tempfile.gettempdir(), rand_prefix)

        print('Generating topic model.')
        mallet_corpus = prefix + 'corpus'
        os.makedirs(mallet_corpus)
        corpus.export(mallet_corpus, abstract=False, form='text')
        model = Mallet(MALLET_PATH, mallet_corpus, num_topics=args.num_topics,
                       iters=args.num_iterations, bigrams=args.bigrams_only, prefix=pre)
    return model

def run_multicore(args, corpus, dictionary, passes, alpha, workers, pre):
    lda = gensim.models.ldamulticore.LdaMulticore
    ldamodel = lda(corpus, num_topics=args.num_topics,
                   id2word=dictionary, passes=200, alpha=20, workers=8,
                   prefix=pre)
    return model

def run_dtm(args, corpus, dictionary, time_slices, pre):
    DTM_PATH = os.environ.get('DTM_PATH', None)
    if not DTM_PATH:
        raise ValueError("You need to set the DTM path.")
    # Run the model
    model = DtmModel(DTM_PATH, corpus=corpus, num_topics=args.num_topics,
        id2word=dictionary, time_slices=time_slices, prefix=pre,
        lda_sequence_max_iter=args.num_iterations)
    return model

def run_ldaseq(args, corpus, dictionary, time_slices):
    # Run the model
    model = LdaSeqModel(corpus=corpus, num_topics=args.num_topics,
        id2word=dictionary, time_slice=time_slices,
        lda_inference_max_iter=args.num_iterations)
    return model

def pylda_vis(args, model, corpus, time_slices, pre):
    print(timestamp() + " About to visualize...", file=sys.stderr)
    for slice in range(len(time_slices)):
        doc_topic, topic_term, doc_lengths, term_frequency, vocab = model.dtm_vis(time=slice, corpus=corpus)
        vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term,
                                       doc_topic_dists=doc_topic,
                                       doc_lengths=doc_lengths,
                                       vocab=vocab,
                                       term_frequency=term_frequency,
                                       sort_topics=True)
        pyLDAvis.save_html(vis_wrapper, pre + "time_slice_" + str(slice) + ".html")
        print(timestamp() + " Prepared time slice", slice, "for pyLDAvis...", file=sys.stderr)

def model_for_year(args, year, files, pre, time_slices):

    # if not args.mixed_ngrams:
    #     # Filter extremes if not doing only bigrams
    #     dictionary.filter_extremes(no_below=50, no_above=0.90)

    if args.gensim or args.model_type != "lda":
        texts = compile_tokens(args, files)

        print(timestamp() + " Building dictionary.", file=sys.stderr)

        dictionary = corpora.Dictionary(texts)
        print(timestamp() + " Reading corpus.", file=sys.stderr)
        corpus = [dictionary.doc2bow(text) for text in texts]
    else:
        print(timestamp() + " Reading corpus.", file=sys.stderr)
        corpus = Corpus(args.corpus_dir)

    # Run the specified model
    if args.model_type == "multicore":
        model = run_multicore(args, corpus, dictionary, 200, 20, 8, pre)
    elif args.model_type == "lda" and args.gensim:
        model = run_lda(args, corpus, pre, dictionary=dictionary, workers=12)
    elif args.model_type == "lda":
        model = run_lda(args, corpus, pre)
    else:
        if args.model_type == "dtm": # Dynamic Topic Model
            model = run_dtm(args, corpus, dictionary, time_slices, pre)
        elif args.model_type == "ldaseq":
            model = run_ldaseq(args, corpus, dictionary, time_slices)

        if args.vis:
            pylda_vis(args, model, corpus, time_slices, pre)

    if not args.gensim:
        return model

    save_model_files(pre, year, model, files)
    if args.model_type == "ldaseq":
        top_words = []
        for t in range(len(time_slices)):
            top_words.append(model.print_topics(time=t, top_terms=20))
        return top_words
    return model.print_topics(num_topics=-1, num_words=20)

def save_model_files(pre, year, model, files):
    append = "" if not year else "-" + str(year)
    # Save model with timestamp
    model.save(pre + "model" + append)

    f = open(pre + "file_ordering" + append + ".txt", "w")
    text = ""
    for filename in files:
        text += filename + " "
    f.write(text[:-1])
    f.close()

def model_on_directory(args):
    # Types of valid models to run
    type_list = ["lda", "multicore", "dtm", "ldaseq"]
    if args.model_type not in type_list:
        print("\"" + args.model_type + "\" is not a valid model type. Please specify a valid model type (" + ", ".join(type_list) + ").", sys.stderr)
        sys.exit(1)

    # Prefix for running lda (modify if files should go to a different directory)
    # FORMAT: "/work/clambert/models/model_type/YYYYMMDD/HH-MM-SS"
    suffix = "-" + args.suffix if args.suffix else ""
    pre = args.save_model_dir + args.model_type + "/" + time.strftime("%Y%m%d") + "/" + time.strftime("%H-%M-%S") + suffix +"/"

    if not os.path.exists(pre):
        os.makedirs(pre)

    print(timestamp() + " Model(s) will be saved to", pre, file=sys.stderr)
    print_params(pre, args)

    print(timestamp() + " Processing corpus.", file=sys.stderr)
    files_dict, time_slices = order_files(args)
    print(timestamp() + " Time slices:", time_slices)
    # Loop for some model types
    if args.model_type in ["lda", "multicore"]:
        for year, files in files_dict.items():
            if len(files_dict.items()) == 1:
                year = ""
                temp_pre = pre
            else:
                temp_pre = os.path.join(pre, str(year) + "/")
            if not os.path.exists(temp_pre):
                os.makedirs(temp_pre)
            print(model_for_year(args, year, files, temp_pre, time_slices))
    # Dynamic models only need to be run once
    elif args.model_type in ["dtm", "ldaseq"]:
        files = []
        for year, file_list in files_dict.items():
            files += file_list

        model_for_year(args, None, files, pre, time_slices)
    print(timestamp() + " Done.", file=sys.stderr)

# _________________________________________________________________________

def main(args):
    arg_dict = vars(args)
    print(timestamp() + " Starting...")
    print(str(arg_dict))
    model_on_directory(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_dir', type=str, default="/work/clambert/models/", help='base directory for saving model directory')
    parser.add_argument('--unigrams_only', default=False, action="store_true", help='whether or not to only include unigrams')
    parser.add_argument('--bigrams_only', default=False, action="store_true", help='whether or not to only include bigrams')
    parser.add_argument('--mixed_ngrams', default=False, action="store_true", help='whether or not to include both unigrams and bigrams')
    parser.add_argument('--corpus_dir', type=str, default="/work/clambert/thesis-data/sessionsPapers-txt-tok", help='directory containing corpus')
    parser.add_argument('--model_type', type=str, default="lda", help='type of model to run') # Include dynamic here?
    parser.add_argument('--num_topics', type=int, default=100, help='number of topics to find')
    parser.add_argument('--optimize_interval', type=int, default=10, help='number of topics to find')
    parser.add_argument('--num_iterations', type=int, default=1000, help='number of topics to find')
    parser.add_argument('--year_split', type=int, default=100, help='Number of years per time slice')
    parser.add_argument('--vis', default=False, action='store_true', help='whether or not to visualize')
    parser.add_argument('--gensim', default=False, action='store_true', help='whether or not to use gensim\'s lda mallet wrapper')
    parser.add_argument('--seed', type=int, default=0, help='random seed to make deterministic')
    parser.add_argument('--suffix', type=str, default="", help="suffix to add to model directory if exists")
    args = parser.parse_args()
    main(args)
