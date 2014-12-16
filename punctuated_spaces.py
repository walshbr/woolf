#!/usr/bin/env python3
# coding: utf-8


import codecs
import collections
import itertools
import os
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np


CORPUS = 'corpus'


def read_text(filename):
    """Read in the text from the file."""
    with codecs.open(filename, 'r', 'utf8') as f:
        return f.read()


def clean_text(input_text):
    """Clean the text by lowercasing and removing newlines."""
    return input_text.replace('\n', ' ').lower()


def find_quoted_quotes(input_text):
    """This returns the regex matches from finding the quoted quotes."""
    return list(re.finditer(r'"[^"]+"', input_text))


def create_location_histogram(file, bin_count=500):
    """\
    This takes the regex matches and produces a histogram of where they
    occurred in the document.
    """
    text = clean_text(read_text(file))
    matches = find_quoted_quotes(text)
    locations = [m.start() for m in matches]
    n, bins = np.histogram(locations, bin_count)

    fig, ax = plt.subplots()

    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    barpath = path.Path.make_compound_path_from_polys(XY)

    patch = patches.PathPatch(
        barpath, facecolor='blue', edgecolor='gray', alpha=0.8,
        )
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.show()


def take_while(pred, input_str):
    """This returns the prefix of a string that matches pred,
    and the suffix where the match stops."""
    for (i, c) in enumerate(input_str):
        if not pred(c):
            return (input_str[:i], input_str[i:])
    else:
        return (input_str, "")


def is_punct(c):
    """Since `unicode` doesn't have a punctuation predicate..."""
    return unicodedata.category(c)[0] == 'P'


def get_unicode_category(unichars, prefix):
    """\
    This returns a generator over the unicode characters with a given major
    category.
    """
    return (c for c in unichars if unicodedata.category(c)[0] == prefix)


def make_token_re():
    unichars = [chr(c) for c in range(sys.maxunicode)]
    punct_chars = re.escape(''.join(get_unicode_category(unichars, 'P')))
    word_chars = re.escape(''.join(get_unicode_category(unichars, 'L')))
    number_chars = re.escape(''.join(get_unicode_category(unichars, 'N')))

    re_token = re.compile(r'''
            (?P<punct>  [{}]  ) |
            (?P<word>   [{}]+ ) |
            (?P<number> [{}]+ ) |
            (?P<trash>  .     )
        '''.format(punct_chars, word_chars, number_chars),
        re.VERBOSE,
        )
    return re_token


def tokenize(input_str, token_re=make_token_re()):
    """This returns an iterator over the tokens in the string."""
    return (
        m.group() for m in token_re.finditer(input_str) if not m.group('trash')
        )


class VectorSpace(object):
    """\
    This manages creating a vector space model of a corpus of documents. It
    makes sure that the indexes are consistent.

    Vectors of numpy arrays.
    """

    def __init__(self):
        self.by_index = {}
        self.by_token = {}

    def __len__(self):
        return len(self.by_index)

    def get_index(self, token):
        """If it doesn't have an index for the token, create one."""
        try:
            i = self.by_token[token]
        except KeyError:
            i = len(self.by_token)
            self.by_token[token] = i
            self.by_index[i] = token
        return i

    def lookup_token(self, i):
        """Returns None if there is no token at that position."""
        return self.by_index.get(i)

    def lookup_index(self, token):
        """Returns None if there is no index for that token."""
        return self.by_token.get(token)

    def vectorize(self, token_seq):
        """This turns a list of tokens into a numpy array."""
        v = [0] * len(self.by_token)
        for token in token_seq:
            i = self.get_index(token)
            if i < len(v):
                v[i] += 1
            elif i == len(v):
                v.append(1)
            else:
                raise Exception(
                    "Invalid index {} (len = {})".format(i, len(v)),
                    )
        return np.array(v)

    def get(self, vector, key):
        """This looks up the key in the vector given."""
        return vector[self.lookup_index(key)]

    def pad(self, array):
        """\
        This pads a numpy array to match the dimensions of this vector space.
        """
        padding = np.zeros(len(self) - len(array))
        return np.concatenate((array, padding))

    def vectorize_corpus(self, corpus):
        """\
        This converts a corpus (tokenized documents) into a collection of
        vectors.
        """
        vectors = [self.vectorize(doc) for doc in corpus]
        vectors = [self.pad(doc) for doc in vectors]
        return vectors


def frequencies(corpus):
    """This takes a list of list of tokens and returns a `Counter`."""
    return collections.Counter(
        itertools.ifilter(lambda t: not (len(t) == 1 and is_punct(t)),
                          itertools.chain.from_iterable(corpus)))


def find_quotes(doc, start_quote='“', end_quote='”'):
    """\
    This takes a tokenized document (with punctuation maintained) and returns
    tuple pairs of the beginning and ending indexes of the quoted quotes.
    """
    start = 0
    while start <= len(doc):
        try:
            start_quote_pos = doc.index(start_quote, start)
            end_quote_pos = doc.index(end_quote, start_quote_pos + 1)
        except ValueError:
            return
        yield (start_quote_pos, end_quote_pos + 1)
        start = end_quote_pos + 1


def tokenize_file(filename):
    text = clean_text(read_text(filename))
    return list(tokenize(text))

def pause():
    """\
    Pauses between each text when processing groups of texts together
    for debugging, mostly, but also to analyze output.
    """
    input("Paused. Type any key to continue.")

def number_of_quotes():
    for (root, _, files) in os.walk(CORPUS):
        for fn in files:
            text = clean_text(read_text(os.path.join(root, fn)))
            matches = find_quoted_quotes(text)
            count = len(matches)
            print("Number of quoted sentences in {}: {}".format(fn, count))   
def main():
    counter = 0
    for (root, _, files) in os.walk(CORPUS):
        for fn in files:
            # print('{}\n{}\n\n'.format(fn, '=' * len(fn)))
            # create_location_histogram(os.path.join(root, fn))

            # tokens = tokenize_file(os.path.join(root, fn))
            # for (start, end) in find_quotes(tokens, '"', '"'):
            #     quote = ' '.join(tokens[start:end])
            #     print('{},{}: {}'.format(start, end, quote))
            number_of_quotes()
            print('\n')
            pause()

# i want it to output a list of the number of quoted sentences in each text.
# find_quoted_quotes produces an array of the quoted sentences. So len(that array) gives me what I want.
if __name__ == '__main__':
    main()
