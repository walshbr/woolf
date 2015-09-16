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
        if len(window) == window_size:
            yield list(window)


def is_quote(tagged_token):
    """Is the tagged token a double-quote character?"""
    (word, _) = tagged_token
    return word in {"''", "``", '"', "^"}


def is_word(tagged_token, _):
    """Is the target a word? This ignores the context of the token."""
    return tagged_token[0].isalnum()


def quote_features(tagged_tokens, is_target=is_verb, is_context=is_quote,
                   amount_of_context=5, feature_history=0):
    """This returns a conditional frequency distribution for tagged
    tokens for which is_target returns True, with the context being
    determined by is_context and the amount_of_context."""

    # TODO: This chunk may be ignoring the outmost items.

    # TODO: Break feature creation into a new function so it can be called when
    # we don't know the answers.

    # TODO: Training on transition points that happen between words ignoring
    # punctuation.

    for chunk in windows(tagged_tokens, 2 * amount_of_context):
        current = chunk[amount_of_context]
        if is_target(current, chunk):
            in_context = any(is_context(t) for t in chunk)
            features = {}
            for offset in range(feature_history+1):
                history = chunk[amount_of_context - offset]
                features['token{}'.format(offset)] = history[0]
                features['tag{}'.format(offset)] = history[1]
            print(in_context, features)
            yield (features, in_context)


def main():
    """The main function."""
    tokens = tokenize_corpus(TAGGED)
    tagger = build_trainer(brown.tagged_sents(categories='news'))
    tagged_tokens = tagger.tag(tokens)

    # Identifying the features.
    training_features = list(quote_features(
        tagged_tokens,
        is_target=is_word,
        feature_history=4,
        ))

    # Dividing features into test and training sets.
    # TODO: Add random shuffle back in.
    # random.shuffle(training_features)
    test_size = int(TEST_SET_RATIO * len(training_features))
    test_set = training_features[:test_size]
    training_set = training_features[test_size:]

    # stay classy
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(nltk.classify.accuracy(classifier, test_set))

    # TODO: We need a baseline of what the accuracy would be tagging nothing in
    # context.

    # TODO: cross-validate.
    # TODO: MOAR TRAINING!


if __name__ == '__main__':
    main()
