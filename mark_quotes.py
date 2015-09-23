#!/usr/bin/env python


"""This takes a classifier and an input document and marks the quotes
in it, based on the classifier."""


import argparse
import pickle
import sys

import nltk

import train_quotes


def get_training_features(tokens):
    return train_quotes.get_training_features(
        tokens,
        is_target=train_quotes.is_word,
        feature_history=2,
    )


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-i', '--input', dest='input', action='store',
                        metavar='INPUT_FILE',
                        help='The input document to mark with quotes.')
    parser.add_argument('-c', '--classifier', dest='classifier',
                        action='store', metavar='PICKLE_FILE',
                        help='The classifier to use marking the quotes.')
    parser.add_argument('-o', '--output', dest='output', metavar='OUTPUT_FILE',
                        help='The file to write the output sentences to.')

    return parser.parse_args(argv)


def main():
    args = parse_args()

    with open(args.classifier, 'rb') as fin:
        classifier = pickle.load(fin)

    with open(args.output, 'w') as fout:
        for sent_tokens in train_quotes.tokenize_corpus(args.input):
            buffer = []
            fsets = get_training_features(sent_tokens)

            for feature in fsets:
                buffer.append(feature['token0'])
                if classifier.classify(feature):
                    buffer.append("^")

            fout.write(' '.join(buffer) + '\n')


if __name__ == '__main__':
    main()
