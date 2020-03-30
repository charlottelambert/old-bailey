#!/usr/bin/env python3
import sys, os, argparse
from vis_topic_mallet import get_topics, read_weighted_keys
# It's possible to automatically align the topics from the different models, and that might be worth doing. Since every word is in every topic (even though its weight might be 0), you can compute the difference between topics by doing for each word w:

def get_diffs(t1_list, t2_list, slices):
    print("Comparing topic differences...", end=' ')
    diffs = {}
    for id1, t1 in t1_list.items():
        # Dictionary mapping words to weights in this topic
        t1_dict = {pair[0]:pair[1] for pair in t1}
        # List of words in this topic
        t1_words = list(t1_dict.keys())

        for id2, t2 in t2_list.items():
            # Dictionary mapping words to weights in this topic
            t2_dict = {pair[0]:pair[1] for pair in t2}
            # List of words in this topic
            t2_words = list(t2_dict.keys())

            key = (str(id1) + "_" + slices[0], str(id2)+ "_" + slices[1])

            # Only go through the words present in these two topics
            mini_vocab = set(list(t1_words + t2_words))
            diffs[key] = 0
            for word in mini_vocab:
                try:
                    t1_val = t1_dict[word]
                except KeyError:
                    t1_val = 0
                try:
                    t2_val = t2_dict[word]
                except KeyError:
                    t2_val = 0

                diffs[key] += abs(t1_val - t2_val)

    return diffs

def compare_topics(n_most, n_least, triples, topic_lists):
    print("Words in the", n_most, "most similar topics:")
    print("-"*50)
    seen_list = {0:[], 1:[], 2:[]}
    acc = 0
    for i in range(n_most):
        skip = False
        triple, score = triples[i]
        for j in range(len(triple)):
            if triple[j] == "-1": continue
            if triple[j] in seen_list[j]:
                triples[i] = -1
                skip = True
                break
        if skip: continue
        else:
            for j in range(len(triple)): seen_list[j].append(triple[j])

        print(str(acc) + ": Topic similarity between:", str(triple) + "; difference:", score, "\n")
        print_similarities(triple, topic_lists)
        acc += 1

        print("-"*50)

    print("Words in the", n_least, "least similar topics:")
    print("-"*50)

    for i in range( -n_least, 0, 1):
        triple, score = triples[i]
        print(str(i) + ": Topic similarity between:", str(triple) + "; difference:", score, "\n")

        print_similarities(triple, topic_lists)
        print("-"*50)

def print_similarities(triple, topic_lists):
    for time_slice, id in enumerate(triple):
        if id == "-1":
            print("No topic for time slice", time_slice)
            continue
        topic = int(id.split("_")[0])
        words = topic_lists[time_slice][topic]

        topic_list = ", ".join([pair[0] for pair in words])
        print("Words in time slice", time_slice, "; topic", str(topic) + ":", topic_list)
        print("-"*50)


def main(args):
    # Extract topics from each model
    print("Getting topics...", file=sys.stderr, end=' ')
    # Open weighted key files
    weighted_keys1 = open(args.weighted_keys_1, 'r')
    weighted_keys2 = open(args.weighted_keys_2, 'r')
    weighted_keys3 = open(args.weighted_keys_3, 'r')
    weighted_keys = [weighted_keys1, weighted_keys2, weighted_keys3]
    # Extract topics and weights
    topic_lists = []
    for wk in weighted_keys:
        topic_lists.append(read_weighted_keys(wk))
    # Close open files
    for wk in weighted_keys:
        wk.close()
    len_list = [str(len(t_list)) for t_list in topic_lists]
    print("Done!", file=sys.stderr)
    print("Number of topics in each time slice:", " ".join(len_list))

    # Collect diffs for each pair of time slices
    diff_dict = {(0,1):{}, (1,2):{}, (0,2):{}}
    for i, j in diff_dict.keys():
        diff_dict[(i,j)] = get_diffs(topic_lists[i], topic_lists[j], (str(i),str(j)))

    print("Done!", file=sys.stderr)
    print("="*50)

    print("Ranking topics by similarity...", file=sys.stderr, end=' ')
    for k, v in diff_dict.items():
        diff_dict[k] = sorted(v.items(), key =
                     lambda kv:(kv[1], kv[0]))
    print("Done!", file=sys.stderr)

    print("-"*50)
    topic_alignments = {}
    for pair, sorted_diffs in diff_dict.items():
        new_alignments_1 = []
        new_alignments_2 = []
        order = []
        # Remove any topic pairs that include topics already assigned a most similar topic
        for topic_pair, similarity in sorted_diffs:
            if topic_pair[0] in new_alignments_1: continue
            if topic_pair[1] in new_alignments_2: continue
            order.append((topic_pair, similarity))
            new_alignments_1.append(topic_pair[0])
            new_alignments_2.append(topic_pair[1])
        topic_alignments[pair] = order

    all_alignments = []
    for label, alignments in topic_alignments.items():
        all_alignments += alignments

    nested = {item[0]:{} for item, score in all_alignments}
    for topic_pair, score in all_alignments:
        nested[topic_pair[0]].update({topic_pair[1]:score})

    triples = {}
    seen_list = {0:[], 1:[], 2:[]}
    for head_topic, matches in nested.items():
        if head_topic in seen_list[0]: continue

        if len(matches) == 2:
            # Look at both matches
            for match, score in matches.items():
                if match in seen_list[1]: continue
                # If first match also has entry in nested
                if match in nested.keys():
                    third = list(nested[match].keys())[0]
                    if third in seen_list[2]: continue
                    # Want the value of match to be the other match in matches
                    if third in matches:
                        triple = (head_topic, match, third)
                        triple_score = score + nested[match][third] + matches[third]
                        triples[triple] = triple_score
                        for j in range(len(triple)): seen_list[j].append(triple[j])
                        break
                else: continue
        else:
            item = list(matches.keys())[0]
            head_posn = int(head_topic.split("_")[1])
            item_posn = int(item.split("_")[1])
            if item in seen_list[item_posn]: continue
            triple = ["-1"]*3
            triple[head_posn] = head_topic
            triple[item_posn] = item
            triples[tuple(triple)] = matches[item]
            seen_list[head_posn].append(head_topic)
            seen_list[item_posn].append(item)

    triples = sorted(triples.items(), key=lambda item: item[1])
    print("Comparing time slice", pair[0], "and time slice", pair[1])
    n_most = 300
    n_least = 0
    compare_topics(n_most, n_least, triples, topic_lists)
    print("-"*50)


    print("\n".join([str(item) for item in triples]))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weighted_keys_1', type=str, help='path to first weighted_keys.txt file to load')
    parser.add_argument('weighted_keys_3', type=str, help='path to second weighted_keys.txt file to load')
    parser.add_argument('weighted_keys_2', type=str, help='path to third weighted_keys.txt file to load')
    args = parser.parse_args()
    main(args)
