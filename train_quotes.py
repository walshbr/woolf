#!/usr/bin/env python3


"""This compares a number of classifiers on a corpus while looking for
"silent" quotations marked in the training corpus with ^."""

# TODO: instead of looking for quote insertions, look for whether the
# current word is in a quote, and include that in the previous words'
# data in the feature set (for each word, look at the token, the tag,
# and the quote).

# TODO: Include a raw version of the corpus (maybe including
# whitespace) and ignore for training, classifying, but use when
# re-creating the input (with quotes inserted).

# TODO: Want to reproduce the histogram of quotation marks with the
# "silent" quotes.


import argparse
import csv
import itertools
from multiprocessing.pool import Pool
import operator
import os
import pickle
import random
import statistics
import sys

import nltk
import nltk.corpus

from fset_manager import Current, TAGGED


TEST_SET_RATIO = 0.2


first = operator.itemgetter(0)
second = operator.itemgetter(1)


def is_verb(token_tag):
    """This returns True if the tagged word is any form of verb, but
    it ignores the rest of the context (the second parameter)."""
    _, tag = token_tag
    return tag.startswith('VB')


def is_quote(token_tag):
    """Is the tagged token a double-quote character?"""
    token, _ = token_tag
    return token in {"''", "``", '"', "^"}


def is_word(context):
    """Is the target a word? This ignores the context of the token."""
    return context.current.token.isalnum()


def produce_confusion_matrix(test_features, classifier):
    """Produces a confusion matrix for the test classifier"""

    gold = [feature for (__, feature) in test_features]
    test = [classifier.classify(features) for (features, __) in test_features]
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


def cross_validate(cls, training_features, num_folds=10):
    """Takes a set of classifier builder, training features, trains a
    classifier based on it, and cross validates it against a specified
    number of folds. Prints out the average accuracy for the
    classifier across num_folds as well as the individual accuracies
    for the subsections."""
    print('Cross validating {}'.format(cls.__name__))
    accuracies = []
    subset_size = int(len(training_features) / num_folds)
    for i in range(num_folds):

        accuracy = 0
        testing_this_round = training_features[i*subset_size:][:subset_size]
        training_this_round = (training_features[:i*subset_size] +
                               training_features[(i+1)*subset_size:])
        classifier = cls.train(training_this_round)
        accuracy = nltk.classify.accuracy(classifier, testing_this_round)
        accuracies.append(accuracy)
        print('Accuracy for fold {} = {}'.format(i, accuracy))

    average = sum(accuracies) / num_folds

    print('Cross-validated accuracy = {}'.format(average))
    return average


def cross_validate_sets(cls, training_features, num_folds=10):
    """Takes a set of classifier builder, training features, trains a
    classifier based on it, and cross validates it against a specified
    number of folds. It yields the classifier class and accuracy."""
    subset_size = int(len(training_features) / num_folds)
    for i in range(num_folds):
        testing_this_round = training_features[i*subset_size:][:subset_size]
        training_this_round = (training_features[:i*subset_size] +
                               training_features[(i+1)*subset_size:])
        yield (cls, training_this_round, testing_this_round)


def cross_validate_p(cls, training, test):
    """This performs the cross-validation on one fold."""
    classifier = cls.train(training)
    accuracy = nltk.classify.accuracy(classifier, test)
    return (cls, accuracy)


def cross_validate_means(accuracies):
    """This takes the means output from cross_validate_p, groups them
    by class, and averages them. It yields the classes and averages."""
    accuracies = list(accuracies)
    accuracies.sort(key=lambda x: first(x).__name__)
    for (cls, accuracy) in itertools.groupby(accuracies, first):
        yield (cls, statistics.mean(x for (_, x) in accuracy))


def get_sets(featuresets, ratio):
    """This breaks a sequence of feature sets into two groups based on
    the ratio."""
    test_size = int(ratio * len(featuresets))
    test_set = featuresets[:test_size]
    training_set = featuresets[test_size:]
    return (test_set, training_set)


def get_baseline(cls, training, test, base_value):
    """This returns the accuracy for a baseline training, i.e.,
    training based on everything being `base_value`."""
    baseline = [(fs, base_value) for (fs, _) in training]
    classifier = cls.train(baseline)
    return nltk.classify.accuracy(classifier, test)


def report_classifier(cls, accuracy, training, test, featureset, outdir, corpus_dir):
    """This reports on a classifier, comparing it to a baseline, and
    pickling it into a directory."""
    name = cls.__name__
    if Current.__name__ == 'InternalStyle':
        model = 'internal'
    else:
        model = 'external'

    # if the appropriate nested folder structure doesn't exist, go ahead and make them so that you can load the classifiers into them.
    if not os.path.exists(os.path.join(outdir, model)):
        os.makedirs(os.path.join(outdir, model))

    if corpus_dir == 'training_passages/tagged_text':
        corpus_dir = 'tagged'
    if not os.path.exists(os.path.join(outdir, model, corpus_dir)):
        os.makedirs(os.path.join(outdir, model, corpus_dir))
    output = os.path.join(outdir, model, corpus_dir, name + '.pickle')
    baseline = get_baseline(cls, training, test, False)
    classifier = cls.train(featureset)
    with open(output, 'wb') as fout:
        pickle.dump(classifier, fout)
    return (output, accuracy, baseline)


def parse_args(argv=None):
    """This parses the command line."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--corpus', dest='corpus', action='store',
                        default=TAGGED,
                        help='The input directory containing the training '
                             'corpus. Default = {}.'.format(TAGGED))
    parser.add_argument('-r', '--ratio', dest='ratio', type=float,
                        default=TEST_SET_RATIO,
                        help='The ratio of documents to use as a test set. '
                        'Default = {}'.format(TEST_SET_RATIO))
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        action='store', default='classifiers',
                        help='The directory to write the pickled classifiers '
                             'to. Default = ./classifiers/.')

    return parser.parse_args(argv)


def main():
    """The main function."""
    args = parse_args()
    print(args.corpus)
    manager = Current(is_quote, is_word)
    featuresets = manager.get_all_training_features(
        manager.get_tagged_tokens(args.corpus)
    )
    featuresets = [(fs, tag) for (fs, _, tag) in featuresets]
    random.shuffle(featuresets)
    test_set, training_set = get_sets(featuresets, args.ratio)

    classifiers = [
        # nltk.ConditionalExponentialClassifier,
        nltk.DecisionTreeClassifier,
        # nltk.MaxentClassifier,
        # nltk.NaiveBayesClassifier,
        # nltk.PositiveNaiveBayesClassifier,
    ]
    folds = itertools.chain.from_iterable(
        cross_validate_sets(cls, featuresets)
        for cls in classifiers
    )
    with Pool() as pool:
        means = list(cross_validate_means(
            pool.starmap(cross_validate_p, folds, 3),
        ))

    means.sort(key=second)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'results.csv'), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(('Output', 'Accuracy', 'Baseline'))
        writer.writerows(
            report_classifier(cls, a, training_set, test_set, featuresets,
                              args.output_dir, args.corpus)
            for (cls, a) in means
        )

    # TODO: MOAR TRAINING!

if __name__ == '__main__':
    main()
