#!/usr/bin/env python3

import click
import re
import math

def topic_name_html(word_weights, global_min=None, global_max=None):
    def invert_hex(hex_number):
        inverse = hex(abs(int(hex_number, 16) - 255))[2:]
        # If the number is a single digit add a preceding zero
        if len(inverse) == 1:
            inverse = '0'+inverse
        return inverse

    def float_to_color(f):
        val = ('%x' % int(f * 255))
        val = invert_hex(val)
        return '#%s%s%s' % (val[:2], val[:2], val[:2])

    vals = [x[1] for x in word_weights]
    val_max = max(vals)
    val_min = math.sqrt(min(vals) / 2)
    val_diff = float(val_max - val_min)

    ret = ''
    for (y, z) in sorted(word_weights, key=lambda x: x[1],
                         reverse=True)[:20]:
        if global_min and global_max and global_max > global_min:
            global_diff = float(global_max - global_min)
            q = float(z - global_min) / global_diff
        else:
            q = float(z - val_min) / val_diff

        ret += '<span style="color:%s" title="%s%% relevant">%s</span>, ' \
            % (float_to_color(q), int(q * 100), y.replace('_', '&nbsp;'))
    ret = ret[:-2]
    return ret


@click.command()
@click.option('--w2v', is_flag=True, help="If weights are from word2vec neighbors.tsv file.")
@click.argument("weighted_keys", type=click.File())
def main(w2v, weighted_keys):
    global_min = None
    global_max = None
    topic_word_weights = {}
    i = 0
    for line in weighted_keys:
        # Skip header for neighbors.tsv file
        if w2v and i == 0:
            i += 1
            continue
        word_weights = []

        delim = "\t" if w2v else ": "
        topic_id, pairs = line.strip().split(delim)
        if w2v: word_weights.append([topic_id, 1.0])
        # No weight for this word
        if pairs == "UNK": continue

        if w2v: pairs = pairs[1:-1]
        find_str = "\(([^,]+), ([^,]+)\),*" if w2v else "\[([^,]+), ([^,]+)\]"
        for pair in re.findall(find_str, pairs):
            t = pair[0] if w2v else pair[1]
            w = pair[1] if w2v else pair[0]
            weight = float(eval(w))
            s = eval(t) if w2v else eval(eval(t))

            word_weights.append([s, weight])
            if not global_min or weight < global_min:
                global_min = weight
            if not global_max or weight > global_max:
                global_max = weight

        topic_word_weights[topic_id] = word_weights
    for topic in topic_word_weights:
        print("<p>" + topic_name_html(topic_word_weights[topic],
                                      global_min=global_min,
                                      global_max=global_max) + "</p>")

if __name__ == "__main__":
    main()
