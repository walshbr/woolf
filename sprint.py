#!/usr/bin/env python3


"""This trains a naive Bayesian classifier on a corpus with "silent" quotations
marked. It then tests that classifier for accuracy and prints that metric
out."""


import nltk
import nltk.corpus
from nltk import wordpunct_tokenize
from nltk.corpus import names
from nltk.corpus import brown
from collections import deque
from math import floor
# import random


TAGGED = 'training_passages/tagged.txt'
TEST_SET_RATIO = 0.2


def tokenize_corpus(filename):
    """Read the corpus into test and training sets."""
    with open(filename) as fin:
        for token in wordpunct_tokenize(fin.read()):
            if token.isalnum():
                yield token
            else:
                for char in token:
                    yield char


def build_trainer(tagged_sents, default_tag='NN'):
    """This builds a tagger from a corpus."""
    name_tagger = [nltk.DefaultTagger('PN').tag(names.words())]

    tagger0 = nltk.DefaultTagger(default_tag)
    tagger1 = nltk.UnigramTagger(tagged_sents, backoff=tagger0)
    tagger2 = nltk.BigramTagger(tagged_sents, backoff=tagger1)
    tagger3 = nltk.UnigramTagger(name_tagger, backoff=tagger2)

    return tagger3


def is_verb(tagged_word, _):
    """This returns True if the tagged word is any form of verb, but
    it ignores the rest of the context (the second parameter)."""
    (_, tag) = tagged_word
    return tag.startswith('VB')


def windows(seq, window_size):
    """This iterates over window_size chunks of seq."""
    window = deque()
    for item in seq:
        window.append(item)
        if len(window) > window_size:
            window.popleft()
        yield list(window)
    while window:
        window.popleft()
        if window:
            yield list(window)


def is_quote(tagged_token):
    """Is the tagged token a double-quote character?"""
    (word, _) = tagged_token
    return word in {"''", "``", '"', "^"}


def is_word(tagged_token, _):
    """Is the target a word? This ignores the context of the token."""
    return tagged_token[0].isalnum()


def get_features(tagged_window, is_target=is_verb, feature_history=0):
    """This returns the feature set for the data in the current window."""

    # TODO: Training on transition points that happen between words ignoring
    # punctuation.

    featureset = None

    index = floor(len(tagged_window) / 2)
    current = tagged_window[index]
    if is_target(current, tagged_window):
        featureset = {}
        for offset in range(feature_history+1):
            (token, tag) = tagged_window[index - offset]
            featureset['token{}'.format(offset)] = token
            featureset['tag{}'.format(offset)] = tag

    return featureset


def get_tag(_, tagged_window, is_context=is_quote):
    """This returns the tag for the feature set to train against. """
    return any(is_context(t) for t in tagged_window)


def get_training_features(tagged_tokens, is_target=is_verb,
                          is_context=is_quote, amount_of_context=5,
                          feature_history=0):
    """This returns a sequence of feature sets and tags to train against for
    the input tokens."""
    window_size = 2 * amount_of_context
    for window in windows(tagged_tokens, window_size):
        if len(window) < window_size:
            continue
        features = get_features(window, is_target, feature_history)
        if features is not None:
            tag = get_tag(features, window, is_context)
            yield (features, tag)


def produce_confusion_matrix(training_features, tagged_tokens, classifier):
    """Produces a confusion matrix for the test classifier"""

    gold = [feature for (__, feature) in training_features]
    test = [classifier.classify(features) for (features, __) in get_training_features(tagged_tokens, is_target=is_word)]
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


def main():
    """The main function."""
    tokens = list(tokenize_corpus(TAGGED))
    tagger = build_trainer(brown.tagged_sents(categories='news'))
    tagged_tokens = tagger.tag(tokens)

    # Identifying the features.
    training_features = list(get_training_features(
        tagged_tokens,
        is_target=is_word,
        feature_history=2,
        ))

    # Dividing features into test and training sets.
    # TODO: Add random shuffle back in.
    # random.shuffle(training_features)
    test_size = int(TEST_SET_RATIO * len(training_features))
    test_set = training_features[:test_size]
    training_set = training_features[test_size:]

    # get a baseline classifier
    baseline_training = [(fs, False) for (fs, _) in training_set]
    baseline = nltk.NaiveBayesClassifier.train(baseline_training)
    print('Baseline = {}'.format(nltk.classify.accuracy(baseline, test_set)))

    # stay classy
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print('Accuracy = {}'.format(nltk.classify.accuracy(classifier, test_set)))

    produce_confusion_matrix(training_features, tagged_tokens, classifier)
    # TODO: output a confusion table
    # links:
    # - http://www.nltk.org/book/ch06.html#confusion-matrices
    # - http://www.nltk.org/_modules/nltk/metrics/confusionmatrix.html
    # - http://www.nltk.org/api/nltk.metrics.html#module-nltk.metrics.confusionmatrix

    # TODO: cross-validate.
    # TODO: MOAR TRAINING!


if __name__ == '__main__':
    main()
