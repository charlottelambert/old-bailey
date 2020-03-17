#!/usr/bin/env python3

import csv
from collections import defaultdict
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

def read_weighted_keys(keyfile):
    word_weights = defaultdict(list)
    r = csv.reader(keyfile, delimiter='\t')
    for row in r:
        topicnum = row[0]
        weighted_words = [(x, float(y)) for (x, y) in
                          zip(row[1::2], row[2::2])]
        word_weights[int(topicnum)] = weighted_words
    return word_weights

@click.command()
@click.argument("weighted_keys", type=click.File())
def main(weighted_keys):
    global_min = None
    global_max = None
    topic_word_weights = read_weighted_keys(weighted_keys)

    for topic in topic_word_weights:
        print("<p>" + topic_name_html(topic_word_weights[topic],
                                      global_min=global_min,
                                      global_max=global_max) + "</p>")

if __name__ == "__main__":
    main()
