#!/usr/bin/env python3
import gensim, sys, argparse
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers.dtmmodel import DtmModel

def main(args):
    f = sys.argv[1]
    #f.seek(0)
    if args.lda_type == "mallet":
        loaded_model = LdaModel.load(args.model)
    elif args.lda_type == "dtm":
        loaded_model = DtmModel.load(args.model)
    else:
        print("Unknown lda_type provided: " + args.lda_type)
        sys.exit(1)

    for topic_num, topic in enumerate(loaded_model.show_topics(num_topics=-1)):

        if args.lda_type == "mallet":
            topic_num, topic_str = topic
        elif args.lda_type == "dtm":
            topic_str = topic


        print(str(topic_num) + ':', end=' ')
        for term in topic_str.split(' + '):
            weight, word = term.split('*')
            if args.lda_type == "dtm":
                word = "\"" + word + "\""
            print(word, end=' ')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model to load')
    parser.add_argument('--lda_type', type=str, default="mallet", help='type of lda that was run') # Include dynamic here?
    parser.add_argument('--num_topics', type=int, default=-1, help='number of topics to print')
    args = parser.parse_args()
    main(args)
