#!/usr/bin/env python3
# coding: utf-8

import sys
import codecs
import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import pandas as pd

from bokeh.charts import Bar, output_file, save
import numpy as np


def parse_args(argv=None):
    """This parses the command line."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--marked-corpus', dest='corpus_folder',
                        action='store', default="""marked_output/marked_corpus/internal/trained_on_tagged/DecisionTreeClassifier/""",
                        help="""The directory containing the marked corpus folder.
                            Default = ./marked_output/marked_corpus/internal/
                            trained_on_tagged/DecisionTreeClassifier/.""")

    parser.add_argument('-u', '--unmarked-corpus',
                        dest='unmarked_corpus_folder',
                        action='store', default='corpus',
                        help="""The directory containing
                        the unmarked corpus to. Default = ./corpus/.""")

    return parser.parse_args(argv)


def all_files(dirname):
    for (root, _, files) in os.walk(dirname):
        for fn in files:
            yield os.path.join(root, fn)


def count_quotation_marks(text):
    return len(list(re.finditer(r'"', text)))


def count_single_quotation_marks(text):
    return len(list(re.finditer(r"'", text)))


def read_text(filename):
    """Read in the text from the file; return a processed text."""
    with codecs.open(filename, 'r', 'utf8') as f:
        return f.read()


def clean_text(input_text):
    """Clean the text by lowercasing and removing newlines."""
    return input_text.replace('\n', ' ').lower()


def clean_and_read_text(input_text):
    return clean_text(read_text(input_text))


def find_carets(text):
    """returns regex matches for the carets in the corpus."""
    return list(re.finditer(r'\^', text))


def find_quote_characters(text):
    """returns matches for quote characters only."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        return list(re.finditer(r'\'', text))
    else:
        return list(re.finditer(r'\"', text))


def find_quoted_quotes(text):
    """This returns the regex matches from finding the quoted
    quotes. Note: if the number of quotation marks is less than fifty
    it assumes that single quotes are used to designate dialogue."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        return list(re.finditer(r'(?<!\w)\'.+?\'(?!\w)', text))
    else:
        return list(re.finditer(r'"[^"]+"', text))


def all_bokeh_graphs(args, marked_corpus, unmarked_corpus,
                     token='compare', bin_count=400):
    for marked_fn in marked_corpus:
        single_bokeh_graph(args, marked_fn, unmarked_corpus,
                           token='compare', bin_count=400)


def single_bokeh_graph(args, marked_fn, unmarked_corpus,
                       token='compare', bin_count=400):

    print(marked_fn)
    unmarked_fn = os.path.basename(marked_fn)
    unmarked_text = clean_and_read_text(args.unmarked_corpus_folder +
                                        '/' + unmarked_fn)
    text = clean_and_read_text(marked_fn)
    if token == 'compare':
        # assumes that you've passed a True, so you're
        # trying to graph comparatively.
        locations, quote_n, bins = find_bin_counts(
            find_quote_characters(unmarked_text), bin_count)
        _, caret_n, _ = find_bin_counts(find_carets(text), bin_count)
        n = quote_n - caret_n
    elif token == 'caret':
        locations, n, bins = find_bin_counts(find_carets(text), bin_count)
    else:
        locations, n, bins = find_bin_counts(
            find_quoted_quotes(unmarked_text), bin_count)

    d_frame = pd.DataFrame(n, columns=['count'])
    output_file('bokeh_graphs/' + re.sub(r'\.txt', '',
                os.path.basename(marked_fn)) + '.html')
    p = Bar(d_frame, legend=False, plot_width=1200)
    p.xaxis.visible = False
    p.xgrid.visible = False
    save(p)


def find_bin_counts(matches, bin_count):
        locations = [m.start() for m in matches]
        n, bins = np.histogram(locations, bin_count)
        return locations, n, bins


def create_location_histogram(args, marked_corpus, unmarked_corpus,
                              token, bin_count=500):
    """\
    This takes the regex matches and produces a histogram of where they
    occurred in the document. Currently does this for all texts in the corpus
    """
    fig, axes = plt.subplots(len(marked_corpus), 1, squeeze=True)
    fig.set_figheight(9.4)
    for (marked_fn, ax) in zip(marked_corpus, axes):
        unmarked_fn = os.path.basename(marked_fn)
        unmarked_text = clean_and_read_text(args.unmarked_corpus_folder +
                                            '/' + unmarked_fn)
        text = clean_and_read_text(marked_fn)
        if token == 'compare':
            # assumes that you've passed a True, so you're
            # trying to graph comparatively.
            locations, quote_n, bins = find_bin_counts(
                find_quote_characters(unmarked_text), bin_count)
            _, caret_n, _ = find_bin_counts(find_carets(text), bin_count)
            n = quote_n - caret_n
        elif token == 'caret':

            locations, n, bins = find_bin_counts(find_carets(text), bin_count)

        else:
            locations, n, bins = find_bin_counts(
                find_quoted_quotes(unmarked_text), bin_count)

        # fig.suptitle(marked_fn, fontsize=14, fontweight='bold')
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
        ax.add_patch(patch)

        # ax.set_xlabel('Position in Text, Measured by Character')
        # ax.set_ylabel('Number of Quotations')

    (base, _) = os.path.splitext(os.path.basename(marked_fn))
    output = os.path.join(args.corpus_folder, base + '.png')
    print('writing to {}'.format(output))
    plt.savefig('results_graphs/' + token, transparent=True)
    plt.show()


def matplot_graph_all_three(args, marked_files, unmarked_files):
    create_location_histogram(args, marked_files, unmarked_files, 'quote')
    create_location_histogram(args, marked_files, unmarked_files, 'caret')
    create_location_histogram(args, marked_files, unmarked_files, 'compare')


def main():
    # NOTE: before any processing you have to clean the text using
    # clean_and_read_text().
    args = parse_args()
    marked_files = list(all_files(args.corpus_folder))
    unmarked_files = list(all_files(args.unmarked_corpus_folder))
    bokeh_graph(args, marked_files, unmarked_files)
    # matplot_graph_all_three(args, marked_files, unmarked_files)

if __name__ == '__main__':
    main()
