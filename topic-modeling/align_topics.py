#!/usr/bin/env python3
import sys, os
from vis_topic_mallet import get_topics, read_weighted_keys
# It's possible to automatically align the topics from the different models, and that might be worth doing. Since every word is in every topic (even though its weight might be 0), you can compute the difference between topics by doing for each word w:

def print_similarities(topic_ids, t1_list, t2_list):
    id1 = int(topic_ids[0].split("_")[0])
    list0 = ", ".join([" ".join(pair[0].split("_")) for pair in t1_list[id1]])

    print("Words in topic", topic_ids[0] + ":", list0, file=sys.stderr)
    print("-"*50, file=sys.stderr)

    id2 = int(topic_ids[1].split("_")[0])
    list1 = ", ".join([" ".join(pair[0].split("_")) for pair in t2_list[id2]])

    print("Words in topic", topic_ids[1] + ":", list1, file=sys.stderr)

def main():
    print("Reading inputs...", file=sys.stderr, end=' ')
    # Load models
    try:
        weighted_keys1 = open(sys.argv[1], 'r')
        weighted_keys2 = open(sys.argv[2], 'r')
    except:
        print("Please input two paths to weighted_keys.txt files.", file=sys.stderr)

    print("Done!", file=sys.stderr)
    # Extract topics from each model
    print("Getting topics...", file=sys.stderr, end=' ')
    # t1_list = get_topics(weighted_keys1, ret_dict=True)
    # t2_list = get_topics(weighted_keys2, ret_dict=True)
    t1_list = read_weighted_keys(weighted_keys1)
    t2_list = read_weighted_keys(weighted_keys2)
    print("Done! Number of topics 1:", len(t1_list), "; Number of topics 2:", len(t2_list), file=sys.stderr)

    print("Comparing topic differences...", file=sys.stderr, end=' ')
    diffs = {}
    for id1, t1 in t1_list.items():
        t1_dict = {pair[0]:pair[1] for pair in t1}
        t1_words = list(t1_dict.keys())

        for id2, t2 in t2_list.items():
            key = (str(id1) + "_1", str(id2) + "_2")
            # Symmetric, avoid computing diffs that exist
            if key in diffs or key[::-1] in diffs:
                continue

            t2_dict = {pair[0]:pair[1] for pair in t2}
            t2_words = list(t2_dict.keys())

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
    print("Done!", file=sys.stderr)
    print("="*50, file=sys.stderr)
    print("Ranking topics by similarity...", file=sys.stderr, end=' ')
    sorted_diffs = sorted(diffs.items(), key =
                 lambda kv:(kv[1], kv[0]))
    print("Done!", file=sys.stderr)

    print("-"*50, file=sys.stderr)
    n_most = 2
    print("Words in the", n_most, "most similar topics:", file=sys.stderr)
    print("-"*50, file=sys.stderr)
    for i in range(n_most):
        sim_ids = sorted_diffs[i][0]
        most_sim_weight = sorted_diffs[i][1]
        print(str(i) + ": Topic similarity between:", str(sim_ids) + "; difference:", most_sim_weight, "\n", file=sys.stderr)

        print_similarities(sim_ids, t1_list, t2_list)

        print("-"*50, file=sys.stderr)
    n_least = n_most
    print("Words in the", n_least, "least similar topics:", file=sys.stderr)
    print("-"*50, file=sys.stderr)

    for i in range( -n_least, 0, 1):
        least_sim_ids = sorted_diffs[-1][0]
        least_sim_weight = sorted_diffs[-1][1]
        print(str(i) + ": Topic similarity between:", str(least_sim_ids) + "; difference:", least_sim_weight, "\n", file=sys.stderr)

        print_similarities(least_sim_ids, t1_list, t2_list)
        print("-"*50, file=sys.stderr)


if __name__ == '__main__':
    main()
# (And since it's symmetric, avoid computing both Diff[T1, T2] and Diff[T2, T1])

# Then each topic is best aligned with the topic it has the lowest difference with.

# But, even if you do this, showing it for every topic and every time slice will be too much. You might want to show as tables, say, the n topics with the best alignment you get across the n time slices (i.e., lowest average difference across time slices) and the highest variation (topics that don't align well across time slices).

# These could be shown as tables where there are, say, 5 rows for the top 5 words and n columns for the n time slices.
