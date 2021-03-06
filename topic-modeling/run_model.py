#!/usr/bin/env python3

###############################################################################
# run_model.py
#
# Run given model on input directory. Either use Mallet LDA, gensim's multicore,
# or DTM based on input.
#
###############################################################################
import sys, os, csv, gensim, time, argparse, pyLDAvis, tempfile, random
from gensim import corpora, models
from gensim.corpora import MalletCorpus, Dictionary
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.models.ldaseqmodel import LdaSeqModel
from mallet import Mallet
from techknacq.corpus import Corpus
from gensim.models.coherencemodel import CoherenceModel
from vis_topic_mallet import get_topics
sys.path.append('../')
from utils import *

def calc_coherence(model, corpus):
    """
        Calculate UMass topic coherence for topics in gensim-run model.

        input:
            model: gensim model
            corpus: corpus associated with model
    """
    cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print(timestamp(),"Topic coherence:", coherence)

def print_params(pre, args):
    """
        Write input parameters to parameters.tsv file.

        input:
            pre (str): path to directory where results are stored
            args (argparse object): input arguments
    """
    # Print out arguments used to file
    with open(pre + "parameters.tsv", "w+") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        arg_dict = vars(args)
        for key in arg_dict:
            tsv_writer.writerow([key, arg_dict[key]])

def compile_tokens(args, files):
    """
        Function to compile tokens from input files.

        input:
            args (argparse object): input arguments
            files (list): list of filepaths to process

        returns list of tokens built from the files in files
    """
    # Compile list of lists of tokens
    texts = []
    print(timestamp() + " Compiling tokens.", file=sys.stderr)
    for i in range(len(files)):
        file = files[i]
        with open(os.path.join(args.corpus_dir, file)) as f:
            text = f.read().lower().replace("\n", " ").split(" ")

            # Changed: Also remove stop words from Mallet version`
            # stop_words = stop.modified_stop_words
            # text = [word for word in text if word not in stop_words]
            texts.append(text)
    return texts

def run_lda(args, corpus, pre, dictionary=None, workers=None, docs=None, num_files=None):
    """
        Function to run LDA using gensim or mallet wrapper code in lda-tools.

        input:
            args (argparse object): input arguments
            corpus: corpus to run model over
            pre (str): path to save all models/etc. to
            dictionary: optional dictionary for the use with gensim model
            workers: optional number of workers used by gensim model
            docs (pair): optional pair of start year and associated lines from
                tsv file for use by mallet when running over time slices
            num_files (int): optional number of files being input

        returns LDA model
    """
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
        mallet_corpus = prefix + 'corpus'

        print('Generating topic model.')
        form = 'tsv' if args.tsv_corpus else "text"
        tsv_corpus = None
        if not args.tsv_corpus:
            os.makedirs(mallet_corpus)
            corpus.export(mallet_corpus, abstract=False, form=form)
        elif args.year_split != -1:
            year, lines = docs
            os.makedirs(mallet_corpus)
            tsv_corpus = os.path.join(mallet_corpus, str(year) + "-tmp.tsv")
            with open(tsv_corpus, 'w') as f:
                f.write("\n".join(lines))
        else:
            tsv_corpus = args.tsv_corpus

        mallet_corpus = None if args.tsv_corpus else mallet_corpus
        model = Mallet(MALLET_PATH, mallet_corpus, num_topics=args.num_topics,
                       iters=args.num_iterations, bigrams=args.bigrams_only,
                       topical_n_grams=args.topical_n_grams,
                       remove_stopwords=(not args.topical_n_grams), prefix=pre,
                       print_output=True, file=tsv_corpus, min_df=args.min_df,
                       max_df=args.max_df, num_files=num_files)
    return model

def run_multicore(args, corpus, dictionary, passes, alpha, workers, pre):
    """
        Function to run LDA using multicore.

        input:
            args (argparse object): input arguments
            corpus: corpus to run LDA over
            dictionary: dictionary from corpus
            passes (int): number of passes to execute
            workers (int): number of workers to use
            pre (str): path to save all results to

        returns multicore LDA model
    """
    lda = gensim.models.ldamulticore.LdaMulticore
    ldamodel = lda(corpus, num_topics=args.num_topics,
                   id2word=dictionary, passes=passes, alpha=alpha, workers=workers,
                   prefix=pre)
    return model

def run_dtm(args, corpus, dictionary, time_slices, pre):
    """
        Function to run DTM over corpus.

        input:
            args (argparse object): input arguments
            corpus: corpus to run LDA over
            dictionary: dictionary from corpus
            time_slices (list): list containing number of files per time time slice
            pre (str): path to save all results to

        returns DTM model
    """
    DTM_PATH = os.environ.get('DTM_PATH', None)
    if not DTM_PATH:
        raise ValueError("You need to set the DTM path.")
    # Run the model
    model = DtmModel(DTM_PATH, corpus=corpus, num_topics=args.num_topics,
        id2word=dictionary, time_slices=time_slices, prefix=pre,
        lda_sequence_max_iter=args.num_iterations)
    return model

def run_ldaseq(args, corpus, dictionary, time_slices):
    """
        Function to run LDASeq model.

        input:
            args (argparse object): input arguments
            corpus: corpus to run LDA over
            dictionary: dictionary from corpus
            time_slices (list): list containing number of files per time time slice

        returns
    """
    # Run the model
    model = LdaSeqModel(corpus=corpus, num_topics=args.num_topics,
        id2word=dictionary, time_slice=time_slices,
        lda_inference_max_iter=args.num_iterations)
    return model

def pylda_vis(args, model, corpus, time_slices, pre):
    """
        Function to visualize model using pyLDAvis

        input:
            args (argparse object): input arguments
            model: LDA model to visualize
            corpus: corpus to run LDA over
            time_slices (list): list containing number of files per time time slice
            pre (str): path to save all results to

        returns
    """
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
    """
        Function to run model over one time slice

        input:
            args (argparse object): input arguments
            year (int): time slice running model over
            files (list): list of files in time slice
            pre (str): path to save all results to
            time_slices (list): list containing number of files per time time slice

        returns printed topics for model run
    """
    if args.gensim or args.model_type != "lda":
        texts = compile_tokens(args, files)

        print(timestamp() + " Building dictionary.", file=sys.stderr)

        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=args.min_df, no_above=args.max_df)

        print(timestamp() + " Reading corpus.", file=sys.stderr)
        corpus = [dictionary.doc2bow(text) for text in texts]
    else:
        dictionary = None
        print(timestamp() + " Reading corpus.", file=sys.stderr)
        path = args.tsv_corpus if args.tsv_corpus else args.corpus_dir
        corpus = Corpus(path, path_list=files)
        num_files = len(files)

    # Run the specified model
    if args.model_type == "multicore":
        model = run_multicore(args, corpus, dictionary, 200, 20, 8, pre)
    elif args.model_type == "lda" and args.gensim:
        model = run_lda(args, corpus, pre, dictionary=dictionary, workers=12)
    elif args.model_type == "lda":
        model = run_lda(args, corpus, pre, docs=(year, files), num_files=num_files)
    else:
        if args.model_type == "dtm": # Dynamic Topic Model
            model = run_dtm(args, corpus, dictionary, time_slices, pre)
        elif args.model_type == "ldaseq":
            model = run_ldaseq(args, corpus, dictionary, time_slices)

        if args.vis:
            pylda_vis(args, model, corpus, time_slices, pre)
    if args.coherence:
        if args.gensim:
            calc_coherence(model, corpus)
    if not args.gensim and args.model_type == "lda":
        return model

    save_model_files(pre, year, model, files)
    if args.model_type == "ldaseq":
        top_words = []
        for t in range(len(time_slices)):
            top_words.append(model.print_topics(time=t, top_terms=20))
        return top_words
    return model.print_topics(num_topics=-1, num_words=20)

def save_model_files(pre, year, model, files):
    """
        Function to save relevant model files for a given model

        input:
            year (int): time slice running model over
            model: model to save files for
            files (list): list of files in time slice

        returns
    """
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
    """
        Function to run model over all files in a directory.

        input:
            args (argparse object): input arguments
    """
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
        for year, docs in files_dict.items():
            if len(files_dict.items()) == 1:
                year = ""
                temp_pre = pre
            else:
                temp_pre = os.path.join(pre, str(year) + "/")
            if not os.path.exists(temp_pre):
                os.makedirs(temp_pre)
            print(model_for_year(args, year, docs, temp_pre, time_slices))

    # Dynamic models only need to be run once
    elif args.model_type in ["dtm", "ldaseq"]:
        docs = []
        for year, file_list in files_dict.items():
            docs += file_list

        model_for_year(args, None, docs, pre, time_slices)
    print(timestamp() + " Done.", file=sys.stderr)

def main(args):
    arg_dict = vars(args)
    print(timestamp() + " Starting...")
    print(str(arg_dict))
    model_on_directory(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_corpus', type=str, default='path to tsv file containing corpus')
    parser.add_argument('--london_lives_file', type=str, default='')
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
    parser.add_argument('--topical_n_grams', default=False, action='store_true', help='whether or not to run topical_n_grams')
    parser.add_argument('--coherence', default=False, action='store_true', help='whether or not to calculate topic coherence')
    parser.add_argument('--min_df', default=0, type=int, help="minimum number of documents needed to contain a word")
    parser.add_argument('--max_df', default=1.0, type=float, help="maximum number of documents allowed to contain a word")
    args = parser.parse_args()
    main(args)
