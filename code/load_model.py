#!/usr/bin/env python3

from gensim.models.ldamodel import LdaModel
import sys

loaded_lda = LdaModel.load(sys.argv[1])
#print(loaded_lda.show_topics(num_topics=-1))

for topic in loaded_lda.show_topics(num_topics=-1):
    topic_num, topic_str = topic
    print(str(topic_num) + ':', end=' ')
    for term in topic_str.split(' + '):
        weight, word = term.split('*')
        print(word, end=' ')
    print()
