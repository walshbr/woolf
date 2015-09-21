#!/usr/bin/env python3


"""This trains a naive Bayesian classifier on a corpus with "silent" quotations
marked. It then tests that classifier for accuracy and prints that metric
out."""


import nltk
import nltk.corpus
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import names
from nltk.corpus import brown
from collections import deque
from math import floor
import random
import pprint


TAGGED = 'training_passages/tagged_text/mrs.dalloway.txt'
TEST_SET_RATIO = 0.2


def tokenize_corpus(filename):
    """Read the corpus a list sentences, each of which is a list of tokens."""
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


def produce_confusion_matrix(test_features, classifier):
    """Produces a confusion matrix for the test classifier"""

    gold = [feature for (__, feature) in test_features]
    test = [classifier.classify(features) for (features, __) in test_features]
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

def cross_validate(training_features, num_folds=10):
    """Takes a set of training features, trains a classifier based on it, and cross validates it against a specified number of folds. Prints out the average accuracy for the classifier across num_folds as well as the individual accuracies for the subsections."""
    accuracies = []
    subset_size = int(len(training_features)/num_folds)
    for i in range(num_folds):
        #this pulls out a chunk for testing and trains on the rest. And it cycles through. So it retrains on each section while testing it against stuff it hasn't seen.
        accuracy = 0
        testing_this_round = training_features[i*subset_size:][:subset_size]
        training_this_round = training_features[:i*subset_size] + training_features[(i+1)*subset_size:]
        classifier = nltk.NaiveBayesClassifier.train(training_this_round)
        accuracy = nltk.classify.accuracy(classifier, testing_this_round)
        accuracies.append(accuracy)
        print('Accuracy for fold {} = {}'.format(i, accuracy))

    average = sum(accuracies)/ num_folds

    print('Cross-validated accuracy = {}'.format(average))

def main():
    """The main function."""
    tagger = build_trainer(brown.tagged_sents(categories='news'))
    tagged_tokens = []
    for sent in tokenize_corpus(TAGGED):
        tagged_tokens.append(tagger.tag(sent))

    # Identifying the features.
    training_features = []
    for sent in tagged_tokens:
        training_features += get_training_features(
            sent, is_target=is_word, feature_history=2,
        )

    test_size = int(TEST_SET_RATIO * len(training_features))
    test_set = training_features[:test_size]
    training_set = training_features[test_size:]

    # Dividing features into test and training sets.
    random.shuffle(training_features)

    # get a baseline classifier
    baseline_training = [(fs, False) for (fs, _) in training_set]
    baseline = nltk.NaiveBayesClassifier.train(baseline_training)
    print('Baseline = {}'.format(nltk.classify.accuracy(baseline, test_set)))

    # # stay classy
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # print('Accuracy = {}'.format(nltk.classify.accuracy(classifier, test_set)))

    produce_confusion_matrix(test_set, classifier)

    # note - the classifier is currently getting rebuilt and trained inside the function. so it's not really being passed something to cross-validate, is it?
    cross_validate(training_features)
    # TODO: MOAR TRAINING!

# question: the way I have things spaced with returns means that, sometimes when this is not the case in the text, two quotes will appear next to each other. If it blasts the line spaces out of existence, it would think that

# It was astonishing that a man of his
# intellect could stoop so low as he did--but that was too harsh a
# phrase--could depend so much as he did upon people's praise.

# "Oh, but," said Lily, "think of his work!"

# the "people's phrase" occurs in the context of a quote, but really that's just an artifact of the way i'm formatting the training data. So that might throw things off. But then, sometimes she DOES punctuate speech with a return before. But in this case I think the artifacts produced by it far exceed the positive examples in the corpus.

if __name__ == '__main__':
    main()
