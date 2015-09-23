#!/usr/bin/env python3


"""This compares a number of classifiers on a corpus while looking for
"silent" quotations marked in the training corpus with ^."""


import argparse
from collections import deque, namedtuple
import random
import operator
import os
import pickle
import sys

import nltk
import nltk.corpus
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import names
from nltk.corpus import brown


TAGGED = 'training_passages/tagged_text/'
TEST_SET_RATIO = 0.2


FeatureContext = namedtuple('FeatureContext',
                            ['history', 'current', 'lookahead'])
TaggedToken = namedtuple('TaggedToken', ['token', 'tag'])


def make_context(window):
    """This makes a FeatureContext from a window of tokens (which will
    become TaggedTokens.)"""
    return FeatureContext(
        [TaggedToken(*t) for t in window[:-2]],
        TaggedToken(*window[-2]),
        TaggedToken(*window[-1]),
    )


def tokenize_corpus(corpus):
    """Read the corpus a list sentences, each of which is a list of tokens."""
    if os.path.isdir(corpus):
        corpus_dir = corpus
        corpus = [
            os.path.join(corpus_dir, fn) for fn in os.listdir(corpus_dir)
        ]
    else:
        corpus = [corpus]
    for filename in corpus:
        with open(filename) as fin:
            for sent in sent_tokenize(fin.read()):
                sent_tokens = []
                for token in wordpunct_tokenize(sent):
                    if token.isalnum():
                        sent_tokens.append(token)
                    else:
                        sent_tokens += token
                yield sent_tokens


def build_trainer(tagged_sents, default_tag='NN'):
    """This builds a tagger from a corpus."""
    name_tagger = [nltk.DefaultTagger('PN').tag(names.words())]

    tagger0 = nltk.DefaultTagger(default_tag)
    tagger1 = nltk.UnigramTagger(tagged_sents, backoff=tagger0)
    tagger2 = nltk.BigramTagger(tagged_sents, backoff=tagger1)
    tagger3 = nltk.UnigramTagger(name_tagger, backoff=tagger2)

    return tagger3


def is_verb(context):
    """This returns True if the tagged word is any form of verb, but
    it ignores the rest of the context (the second parameter)."""
    return context.current.tag.startswith('VB')


def windows(seq, window_size):
    """This iterates over window_size chunks of seq."""
    window = deque()
    for item in seq:
        window.append(item)
        if len(window) > window_size:
            window.popleft()
        yield list(window)


def is_quote(context):
    """Is the tagged token a double-quote character?"""
    return context.lookahead.token in {"''", "``", '"', "^"}


def is_word(context):
    """Is the target a word? This ignores the context of the token."""
    return context.current.token.isalnum()


def get_features(context):
    """This returns the feature set for the data in the current window."""

    featureset = {
        'token0': context.current[0],
        'tag0': context.current[1],
    }
    history = reversed(list(context.history))
    for (offset, (token, tag)) in enumerate(history):
        featureset['token{}'.format(offset+1)] = token
        featureset['tag{}'.format(offset+1)] = tag

    return featureset


def get_tag(_features, context, is_context=is_quote):
    """This returns the tag for the feature set to train against.
    """
    return is_context(context)


def get_training_features(tagged_tokens, is_target=is_verb,
                          is_context=is_quote, feature_history=0):
    """This returns a sequence of feature sets and tags to train against for
    the input tokens."""
    window_size = feature_history + 2
    for window in windows(tagged_tokens, window_size):
        if len(window) < 2:
            continue
        context = make_context(window)
        if is_target(context):
            features = get_features(context)
            tag = get_tag(features, context, is_context)
            yield (features, tag)


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
        # this pulls out a chunk for testing and trains on the
        # rest. And it cycles through. So it retrains on each section
        # while testing it against stuff it hasn't seen.
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


def get_tagged_tokens(corpus):
    """This tokenizes, segments, and tags all the files in a directory."""
    tagger = build_trainer(brown.tagged_sents(categories='news'))
    tagged_tokens = []
    for sent in tokenize_corpus(TAGGED):
        tagged_tokens.append(tagger.tag(sent))
    return tagged_tokens


def get_all_training_features(tagged_tokens):
    """This takes tokenized, segmented, and tagged files and gets
    training features."""
    training_features = []
    for sent in tagged_tokens:
        training_features += get_training_features(
            sent, is_target=is_word, feature_history=2,
        )
    return training_features


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


def report_classifier(cls, accuracy, training, test, featureset, outdir):
    """This reports on a classifier, comparing it to a baseline, and
    pickling it into a directory."""
    name = cls.__name__
    output = os.path.join(outdir, name + '.pickle')
    baseline = get_baseline(cls, training, test, False)
    print('\t'.join([output, accuracy, baseline]))
    classifier = cls.train(featureset)
    with open(output, 'wb') as fout:
        pickle.dump(classifier, fout)


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

    tagged_tokens = get_tagged_tokens(args.corpus)
    featuresets = get_all_training_features(tagged_tokens)
    random.shuffle(featuresets)
    test_set, training_set = get_sets(featuresets, args.ratio)

    # note - the classifier is currently getting rebuilt and trained
    # inside the function. so it's not really being passed something
    # to cross-validate, is it?

    classifiers = [
        nltk.ConditionalExponentialClassifier,
        nltk.DecisionTreeClassifier,
        nltk.MaxentClassifier,
        nltk.NaiveBayesClassifier,
        # nltk.PositiveNaiveBayesClassifier,
    ]
    means = [
        (cls, cross_validate(cls, featuresets)) for cls in classifiers
    ]
    means.sort(key=operator.itemgetter(1))

    os.makedirs(args.output_dir)

    print("Output\tAccuracy\tBaseline")
    for (cls, a) in means:
        report_classifier(cls, a, training_set, test_set, featuresets,
                          args.output_dir)

    # TODO: MOAR TRAINING!

# question: the way I have things spaced with returns means that,
# sometimes when this is not the case in the text, two quotes will
# appear next to each other. If it blasts the line spaces out of
# existence, it would think that

# It was astonishing that a man of his
# intellect could stoop so low as he did--but that was too harsh a
# phrase--could depend so much as he did upon people's praise.

# "Oh, but," said Lily, "think of his work!"

# the "people's phrase" occurs in the context of a quote, but really
# that's just an artifact of the way i'm formatting the training
# data. So that might throw things off. But then, sometimes she DOES
# punctuate speech with a return before. But in this case I think the
# artifacts produced by it far exceed the positive examples in the
# corpus.

if __name__ == '__main__':
    main()
