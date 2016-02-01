#!/usr/bin/env python3


"""\
usage: istyle.py INPUT_DIR
"""

import os
import random
import re
import sys
import itertools
from multiprocessing.pool import Pool

import nltk
from nltk.corpus import brown, names

from ps import all_files
from train_quotes import get_sets


QUOTED = 1
UNQUOTED = 0


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




def main():
    if (len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv or
        'help' in sys.argv):
        print(__doc__)
        sys.exit()

    corpus_dir = sys.argv[1]
    sent_tokens = nltk.load('tokenizers/punkt/{0}.pickle'.format('english'))
    tagger = build_trainer(brown.tagged_sents())
    corpus = []

    print('reading corpus')
    for fn in all_files(corpus_dir):
        with open(fn) as f:
            text = f.read()
        for (tag, chunk) in find_quoted_quotes(text):
            for sent in get_sentences(chunk, sent_tokens, tagger):
                corpus.append((sent, tag))

    # TODO: figure out how we want to handle the feature sets:
    # existence of words or tf-idf?

    random.shuffle(corpus)
    test_set, training_set = get_sets(corpus, 0.2)
    classifiers = [
        nltk.ConditionalExponentialClassifier,
        nltk.DecisionTreeClassifier,
        nltk.MaxentClassifier,
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
    # TODO: pull in/modify cross_validate_sets
    # TODO: run x-validation in a pool
    # TODO: print out and output a la train_quotes
    # TODO: take the output and re-run it on new documents


if __name__ == '__main__':
    main()
