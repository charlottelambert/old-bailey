#!/usr/bin/env python3
import gensim, sys, argparse
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.models.ldaseqmodel import LdaSeqModel

def main(args):
    f = sys.argv[1]
    if args.lda_type == "mallet":
        loaded_model = LdaModel.load(args.model)

        for topic_num, topic in enumerate(loaded_model.show_topics(num_topics=-1)):

            if args.lda_type == "mallet":
                topic_num, topic_str = topic

            print(str(topic_num) + ':', end=' ')
            for term in topic_str.split(' + '):
                weight, word = term.split('*')
                if args.lda_type == "dtm":
                    word = "\"" + word + "\""
                print(word, end=' ')
            print()

    elif args.lda_type == "dtm":
        loaded_model = DtmModel.load(args.model)
        # maybe use dtm_coherence?
        for topic_id in range(loaded_model.num_topics):
            for time in range(len(loaded_model.time_slices)):
                top_words = loaded_model.show_topic(topic_id, time, topn=10)

                print("Topic", str(topic_id) + ", time slice", str(time) + ':', end=' ')
                for weight, word in top_words:
                    print(word, end=' ')
                print()
            print()
    elif args.lda_type == "ldaseq":
        loaded_model = LdaSeqModel.load(args.model)
        # maybe use dtm_coherence?
        print(loaded_model.num_topics)
        print(loaded_model.time_slice)
        for topic_id in range(loaded_model.num_topics):
            for time in range(len(loaded_model.time_slice)):
                top_words = loaded_model.print_topic(topic=topic_id, time=time, top_terms=20)
#                top_words = loaded_model.show_topic(topic_id, time, topn=10)

                print("Topic", str(topic_id) + ", time slice", str(time) + ':', end=' ')
                for word, weight in top_words:
                    print(word, end=' ')
                print()
            print()
    else:
        print("Unknown lda_type provided: " + args.lda_type)
        sys.exit(1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to model to load')
    parser.add_argument('--lda_type', type=str, default="mallet", help='type of lda that was run') # Include dynamic here?
    parser.add_argument('--num_topics', type=int, default=-1, help='number of topics to print')
    args = parser.parse_args()
    main(args)
