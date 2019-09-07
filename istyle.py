#!/usr/bin/env python3


"""\
usage: istyle.py INPUT_DIR
"""

import random
import re
import sys
import itertools
import argparse
from multiprocessing.pool import Pool

import nltk
from nltk.corpus import brown, names

import pickle
import csv
import os
from ps import all_files
from train_quotes import get_sets
import statistics
import operator

QUOTED = 1
UNQUOTED = 0
CORPUS = 'corpus'

TEST_SET_RATIO = 0.2

first = operator.itemgetter(0)
second = operator.itemgetter(1)


def count_quotation_marks(text):
    """"counts the number of double quotation marks in a text"""
    return len(list(re.finditer(r'"', text)))


def count_single_quotation_marks(text):
    """counts number of single quotation marks in a text"""
    return len(list(re.finditer(r"'", text)))


def find_quoted_quotes(text):
    """This returns the regex matches from finding the quoted
    quotes. Note: if the number of quotation marks is less than fifty
    it assumes that single quotes are used to designate dialogue."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        splits = re.split(r'((?<!\w)\'.+?\'(?!\w))', text)
    else:
        splits = re.split(r'("[^"]+")', text)
    for chunk in splits:
        if chunk[0] in ("'", '"'):
            tag = QUOTED
        else:
            tag = UNQUOTED
        yield (tag, chunk)


def build_trainer(tagged_sents, default_tag='DEFAULT'):
    """This builds a tagger from a corpus."""
    name_tagger = [
        nltk.DefaultTagger('PN').tag([
            name.lower() for name in names.words()
        ])
    ]
    punctuation_tags = [[('^', '^'), ('"', '"')]]
    patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*ly$', 'RB'),                       # adverbs
        # comment out the following line to raise to the surface all
        # the words being tagged by this last, default tag when you
        # run debug.py.
        (r'.*', 'NN')                     # nouns (default)
    ]

    # Right now, nothing will get to the default tagger, because the
    # regex taggers last pattern essentially acts as a default tagger,
    # tagging everything as NN.
    tagger0 = nltk.DefaultTagger(default_tag)
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=tagger0)
    punctuation_tagger = nltk.UnigramTagger(
        punctuation_tags, backoff=regexp_tagger
    )
    tagger1 = nltk.UnigramTagger(tagged_sents, backoff=punctuation_tagger)
    tagger2 = nltk.BigramTagger(tagged_sents, backoff=tagger1)
    tagger3 = nltk.UnigramTagger(name_tagger, backoff=tagger2)

    return tagger3


def get_sentences(text, sent_tokenizer, tagger):
    """Yields the sentences in the text. Each token has the normalized
    token text, tag, and its start and ending positions."""
    for start, end in sent_tokenizer.span_tokenize(text):
        sent = text[start:end]
        tokens = []
        pos = []
        matches = re.finditer(
            r'\w+|[\'\"\/^/\,\-\:\.\;\?\!\(0-9]', sent
        )
        for match in matches:
            mstart, mend = match.span()
            tokens.append(match.group(0).lower().replace('_', ''))
            pos.append((mstart+start, mend+start))
        tags = tagger.tag(tokens)
        yield list(zip(tags, pos))


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
    # print('TRAINING', training)
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


def report_classifier(cls, accuracy, training, test, featureset, outdir):
    """This reports on a classifier, comparing it to a baseline, and
    pickling it into a directory."""
    name = cls.__name__
    output = os.path.join(outdir, name + '.pickle')
    baseline = get_baseline(cls, training, test, False)
    classifier = cls.train(featureset)
    with open(output, 'wb') as fout:
        pickle.dump(classifier, fout)
    return (output, accuracy, baseline)


def get_baseline(cls, training, test, base_value):
    """This returns the accuracy for a baseline training, i.e.,
    training based on everything being `base_value`."""
    baseline = [(fs, base_value) for (fs, _) in training]
    classifier = cls.train(baseline)
    return nltk.classify.accuracy(classifier, test)


def get_features(sent):
    """Turn a sentence list into a frequency mapping."""
    return nltk.FreqDist(tok for (tok, _pos) in sent)


def parse_args(argv=None):
    """This parses the command line."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--corpus', dest='corpus', action='store',
                        default=CORPUS,
                        help='The input directory containing the training '
                             'corpus. Default = {}.'.format(CORPUS))
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
    args = parse_args()

    corpus_dir = args.corpus
    sent_tokens = nltk.load('tokenizers/punkt/{0}.pickle'.format('english'))
    tagger = build_trainer(brown.tagged_sents())
    corpus = []

    print('reading corpus')
    for fn in all_files(corpus_dir):
        with open(fn, encoding='latin1') as f:
            text = f.read()
        for (tag, chunk) in find_quoted_quotes(text):
            for sent in get_sentences(chunk, sent_tokens, tagger):
                corpus.append((get_features(sent), tag))

    # TODO: figure out how we want to handle the feature sets:
    # existence of words or tf-idf?

    random.shuffle(corpus)
    test_set, training_set = get_sets(corpus, args.ratio)
    classifiers = [
        # nltk.ConditionalExponentialClassifier,
        # nltk.DecisionTreeClassifier,
        # nltk.MaxentClassifier,
        nltk.NaiveBayesClassifier,
        # nltk.PositiveNaiveBayesClassifier,
    ]

    folds = itertools.chain.from_iterable(
        cross_validate_sets(cls, corpus)
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
            report_classifier(cls, a, training_set, test_set, corpus,
                              args.output_dir)
            for (cls, a) in means
        )

    # TODO: take the output and re-run it on new documents
    # we throw away the positions. we want to hold onto the start and ends of the sentences somehow. put it in a separate file.

if __name__ == '__main__':
    main()
