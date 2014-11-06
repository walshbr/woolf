#!/usr/bin/env python
# coding: utf-8


import codecs
import collections
import itertools
import re
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np


FILENAME = 'Mrs.Dalloway.txt'


def read_text(filename):
    """Read in the text from the file."""
    with codecs.open(filename, 'r', 'utf8') as f:
        return f.read()


def clean_text(input_text):
    """Clean the text by lowercasing and removing newlines."""
    return input_text.replace('\n', '').lower()


def find_quoted_quotes(input_text):
    """This returns the regex matches from finding the quoted quotes."""
    return list(re.finditer(ur'“[^”]+”', input_text))


def create_location_histogram(matches, bin_count=500):
    """\
    This takes the regex matches and produces a histogram of where they
    occurred in the document.
    """
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


def tokenize(input_str):
    """This returns an iterator over the tokens in the string."""
    rest = None

    # Since punctuations are always single characters, this isn't
    # handled by `take_while`.
    if is_punct(input_str[0]):
        yield input_str[0]
        rest = input_str[1:]

    else:
        # Try to match a string of letters or numbers. The first
        # that succeeds, yield the token and stop trying.
        for p in (unicode.isalpha, unicode.isdigit):
            token, rest = take_while(p, input_str)
            if token:
                yield token
                break
        # If it wasn't a letter or number, skip a character.
        else:
            rest = input_str[1:]

    # If there's more to try, get its tokenize and yield them.
    if rest:
        for token in tokenize(rest):
            yield token


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
