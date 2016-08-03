#!/usr/bin/env python3
# coding: utf-8


import codecs
import collections
import itertools
import operator
import os
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


CORPUS = 'marked_output/marked_corpus/internal/trained_on_tagged/DecisionTreeClassifier'
UNMARKED_CORPUS = 'corpus'

def read_text(filename):
    """Read in the text from the file; return a processed text."""
    with codecs.open(filename, 'r', 'utf8') as f:
        return f.read()


def clean_text(input_text):
    """Clean the text by lowercasing and removing newlines."""
    return input_text.replace('\n', ' ').lower()


def clean_and_read_text(input_text):
    return clean_text(read_text(input_text))


def quotations_check(text, file_name):
    """Checks if a file has an even number of quotes."""
    if count_quotation_marks(text) % 2 != 0:
        print("%(file_name)s has an odd number of quotation marks." % locals())
        pause()
    elif count_quotation_marks(text) < 50:
        print("%(file_name)s has a very low number of double quotation marks." % locals())
        count = count_single_quotation_marks(text)
        print("%(file_name)s has %(count)s single quotation marks." % locals() %locals())
        pause()
    elif percent_quoted(text) > 30:
        print("%(file_name)s has a high percentage of quoted text." % locals())
        pause()
    else:
        print("%(file_name)s checks out." % locals())
        pause()


def count_quotation_marks(text):
    return len(list(re.finditer(r'"', text)))


def count_single_quotation_marks(text):
    return len(list(re.finditer(r"'", text)))


def print_long_quotes(text):
    """Iterates over the matches and returns the first one that is
    greater than 100 characters.  Not exact, but it will give a sense
    of when a quotation mark is missing and it starts flagging
    everything as quoted."""
    quotes = find_quoted_quotes(text)
    for idx, match in enumerate(quotes):
        if len(match.group(0)) > 250:
            print("Match %(idx)i:" % locals() + match.group(0))


def print_matches_for_debug(text):
    """Takes a file, finds its matches and prints them out to a new file
    'debug.txt' for debugging."""
    quotes = find_quoted_quotes(text)
    debug = open('debug.txt', 'w')
    counter = 0
    for match in quotes:
        debug.write("Match %(counter)i: " % locals() + match.group(0) + "\n")
        counter += 1
    debug.close()


def find_quoted_quotes(text):
    """This returns the regex matches from finding the quoted
    quotes. Note: if the number of quotation marks is less than fifty
    it assumes that single quotes are used to designate dialogue."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        return list(re.finditer(r'(?<!\w)\'.+?\'(?!\w)', text))
    else:
        return list(re.finditer(r'"[^"]+"', text))

def find_carets(text):
    """returns regex matches for the carets in the corpus."""
    return list(re.finditer(r'^', text))


def split_quoted_quotes(text):
    """This partitions a text into quotes and non-quotes. Note: if the number
    of quotation marks is less than fifty it assumes that single quotes are
    used to designate dialogue."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        return re.split(r'((?<!\w)\'.+?\'(?!\w))', text)
    else:
        return re.split(r'("[^"]+")', text)


def find_bin_counts(matches, bin_count):
        locations = [m.start() for m in matches]
        n, bins = np.histogram(locations, bin_count)
        return locations, n, bins


def create_location_histogram(corpus, unmarked_corpus, compare, bin_count=500):
    """\
    This takes the regex matches and produces a histogram of where they
    occurred in the document. Currently does this for all texts in the corpus
    """

    # subtract locations - Now that you have the counter object, where do you go from there. Is that the right way to subtract them?
    fig, axes = plt.subplots(len(corpus), 1, squeeze=True)
    fig.set_figheight(9.4)
    for (fn, ax) in zip(corpus, axes):
        unmarked_fn = re.sub('.*/', '', fn)
        unmarked_text = clean_and_read_text(UNMARKED_CORPUS + '/' +unmarked_fn)
        text = clean_and_read_text(fn)
        if compare == 'compare':
            # assumes that you've passed a True, so you're trying to graph comparatively.
            locations, quote_n, bins = find_bin_counts(find_quoted_quotes(unmarked_text), bin_count)
            _, caret_n, _ = find_bin_counts(find_carets(text), bin_count)
            n = quote_n - caret_n
        elif compare == 'caret':
            print(fn)
            print(UNMARKED_CORPUS)
            print(unmarked_fn)
            locations, n, bins = find_bin_counts(find_carets(text), bin_count)
        else:
            locations, n, bins = find_bin_counts(find_quoted_quotes(unmarked_text), bin_count)

        # fig.suptitle(fn, fontsize=14, fontweight='bold')
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        top = bottom + n

        XY = np.array(
            [[left, left, right, right], [bottom, top, top, bottom]]
        ).T

        barpath = path.Path.make_compound_path_from_polys(XY)

        patch = patches.PathPatch(
            barpath, facecolor='blue', edgecolor='gray', alpha=0.8,
            )

        ax.set_xlim(left[0], right[-1])
        ax.set_ylim(bottom.min(), top.max())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # plt.axis('off')
        ax.add_patch(patch)

        # ax.set_xlabel('Position in Text, Measured by Character')
        # ax.set_ylabel('Number of Quotations')

    (base, _) = os.path.splitext(os.path.basename(fn))
    output = os.path.join(CORPUS, base + '.png')
    print('writing to {}'.format(output))
    plt.savefig('results_graphs/' + compare, transparent=True)
    plt.show()
    print(compare)


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
    """This takes a list of tokens and returns a `Counter`."""
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


def tokenize_file(text):
    return list(tokenize(text))


def pause():
    """\
    Pauses between each text when processing groups of texts together for
    debugging, mostly, but also to analyze the sometimes really long output.
    """
    input("Paused. Type any key to continue.")


def calc_number_of_quotes(text):
    """returns the number of characters contained in quotation marks"""
    matches = find_quoted_quotes(text)
    text_string = ""
    for match in matches:
        text_string = text_string + match.group(0)
    count = len(text_string)
    return count


def calc_number_of_characters(text):
    text = text.replace('\\', '')
    count = len(text)
    return count


def percent_quoted(text):
    number_of_quotes = calc_number_of_quotes(text)
    number_of_characters = calc_number_of_characters(text)
    percent = 100 * (number_of_quotes / number_of_characters)
    return percent


def average_sentence_length(text):
    """Meant to calculate the average length of a quoted sentence."""
    matches = find_quoted_quotes(text)
    """Note: right those quotations that break in mid-sentence: "What
    is the point," she said, "since all of this happened."  are
    treated as separate sentences.- but they're meant to be part of
    the same chunk. So the data needs massaging.  Need a regex to
    search for period and quotation mark pairings. Also note that it's
    including the quotation marks in its character count."""
    number_of_matches = len(matches)
    number_of_quoted_characters = calc_number_of_quotes(text)
    average_quoted_sentence_length = (number_of_quoted_characters
                                      / number_of_matches)
    return average_quoted_sentence_length


def corpus_list_average_sentence_lengths(corpus):
    print("Average Sentence Lengths in the corpus.")
    print("\n=============\n")
    for fn in corpus:
        text = clean_and_read_text(fn)
        average_length = average_sentence_length(text)
        print("\n=============\n" + fn)
        print("The average sentence length is {}".format(average_length))
        print("=============")


def corpus_list_number_of_quoted_characters(corpus):
    print("Number of quoted characters in the corpus.")
    print("\n=============\n")
    for fn in corpus:
        text = clean_and_read_text(fn)
        number = calc_number_of_quotes(text)
        print("\n=============\n" + fn)
        print("The number of quoted characters is {}".format(number))
        print("=============")


def corpus_list_percentage_quoted(corpus):
    print("Percentage of each text that is quoted material in the corpus.")
    print("\n=============\n")
    for fn in corpus:
        text = clean_and_read_text(fn)
        percent = percent_quoted(text)
        print("\n=============\n" + fn)
        print("The percentage of quoted text is {}".format(percent))
        print("=============")


def print_stats(corpus):
    """prints stats to the terminal. when you implement the csv export
    this will likely be obsolete"""
    corpus_list_percentage_quoted(corpus)
    corpus_list_average_sentence_lengths(corpus)
    corpus_list_number_of_quoted_characters(corpus)


def all_files(dirname):
    for (root, _, files) in os.walk(dirname):
        for fn in files:
            yield os.path.join(root, fn)


def top_items(vectorizer, array, n=10):
    inv_vocab = dict((v, k) for (k, v) in vectorizer.vocabulary_.items())
    for row in array:
        indexes = list(enumerate(row))
        indexes.sort(key=operator.itemgetter(1), reverse=True)
        top = [(i, inv_vocab[i], c) for (i, c) in indexes[:n]]
        yield top


def vectorizer_report(title, klass, filenames, **kwargs):
    params = {
        'input': 'filename',
        'tokenizer': tokenize,
        'stop_words': 'english',
        }
    params.update(kwargs)
    v = klass(**params)
    corpus = v.fit_transform(filenames)
    a = corpus.toarray()

    print('# {}\n'.format(title))
    for (fn, top) in zip(filenames, top_items(v, a)):
        print('## {}\n'.format(fn))
        for row in top:
            print('{0[0]:>6}. {0[1]:<12}\t{0[2]:>5}'.format(row))
        print()


def concatenate_quotes(text):
    quotes = find_quoted_quotes(text)
    counter = 0
    concatenated_quotes = ""
    for match in quotes:
        concatenated_quotes += quotes[counter].group(0)
        counter += 1
    return concatenated_quotes


def main():
    # NOTE: before any processing you have to clean the text using
    # clean_and_read_text().

    marked_files = list(all_files(CORPUS)) 
    unmarked_files = list(all_files(UNMARKED_CORPUS))
    # # remove_short = lambda s: filter(lambda x: len(x) > 1, tokenize(s))
    # # vectorizer_report(
    # #     'Raw Frequencies', CountVectorizer, files, tokenizer=remove_short,
    # #     )
    # # vectorizer_report('Tf-Idf', TfidfVectorizer, files,
    # #                   tokenizer=remove_short)
    # print_stats(files)
    create_location_histogram(marked_files, unmarked_files, 'quote')
    create_location_histogram(marked_files, unmarked_files, 'caret')
    create_location_histogram(marked_files, unmarked_files, 'compare')
if __name__ == '__main__':
    main()

# To do:

# probably a good time to use classes

# get eric to show you how to convert the vectorizer report to work
# only on quoted text.  you can set a processing and preprocessing
# step

# Also make sure, once all the functions are written, that you don't have
# redundant cleaning of texts and looping through the corpus.

# It's currently preserving \s for every quote. Do we want to keep that?
# Presumably? It's going to throw off the percentages though.

# don situation

# have it clean up a text file that it reads in.
# automate it to run over the gutenberg corpus
