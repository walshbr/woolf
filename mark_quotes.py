#!/usr/bin/env python


"""This takes a classifier and an input document and marks the quotes
in it, based on the classifier."""

# TODO: Don't strip out existing punctuation.

# TODO: Only show new quotation marks.

# TODO: get lines from input to the insert quotes function.


import argparse
import pickle
import sys

import train_quotes


def get_training_features(tokens):
    """This wraps `get_training_features` from `train_quotes` with the
    parameters used in training. This should probably be stored
    somewhere."""
    return train_quotes.get_training_features(
        tokens,
        is_target=train_quotes.is_word,
        feature_history=2,
    )


def load_classifier(filename):
    """Loads the classifier from pickled into `filename`."""
    with open(filename, 'rb') as fin:
        return pickle.load(fin)


def insert_quotes(classifier, fsets, sentence):
    """Inserts ^ quotes wherever the classifier says they should be."""
    # print(list(fsets))
    # print(sentence)
    # print("*****")
    for ((token,_), span) in sentence:
        yield token
        for (feature, _) in fsets:
            if token == feature['token0'] and classifier.classify(feature):
                yield "^"            



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
    classifier = load_classifier(args.classifier)


    with open(args.output, 'w') as fout:
        tagged_tokens = [sent for sent in train_quotes.get_tagged_tokens(args.input)]
        for sentence in tagged_tokens:
            tokens = [token for (token, _) in sentence]
            spans = [span for (_, span) in sentence]
            sent = insert_quotes(
                classifier,
                get_training_features(sentence),
                sentence
            )
            fout.write(' '.join(sent) + '\n')


if __name__ == '__main__':
    main()
