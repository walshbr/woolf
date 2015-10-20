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

#B's to dos:

# TODO: Refine the POS tagger so that it catches punctuation appropriately. It's catching a lot of it in the default tagger as NN right now.

# TODO: Class for featureset, tag, and other processing (so we can
# swap them out as a group together)

import argparse
from collections import deque, namedtuple
import csv
import itertools
from multiprocessing.pool import Pool
import operator
import os
import pickle
import random
import statistics
import sys
import re

import nltk
import nltk.corpus
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.corpus import names
from nltk.corpus import brown


TAGGED = 'training_passages/tagged_text/'
TEST_SET_RATIO = 0.2


FeatureContext = namedtuple('FeatureContext',
                            ['history', 'current', 'lookahead'])
TaggedToken = namedtuple('TaggedToken',['token', 'tag', 'start', 'end'])

first = operator.itemgetter(0)
second = operator.itemgetter(1)


# [((TOKEN, TAG), (START, END))] -> FeatureContext
def make_context(window):
    """This makes a FeatureContext from a window of tokens (which will
    become TaggedTokens.)"""
    return FeatureContext(
        [tagged_token(t) for t in window[:-2]],
        tagged_token(window[-2]),
        tagged_token(window[-1]),
    )


def tagged_token(token_span):
    """This takes an input of ((TOKEN, TAG), (START, END)) and returns
    a TaggedToken."""
    ((token, tag), (start, end)) = token_span
    return TaggedToken(token, tag, start, end)


def tokenize_corpus(corpus):
    """Read the corpus a list sentences, each of which is a list of
    tokens and the spans in which they occur in the text."""
    if os.path.isdir(corpus):
        corpus_dir = corpus
        corpus = [
            os.path.join(corpus_dir, fn) for fn in os.listdir(corpus_dir)
        ]
    else:
        corpus = [corpus]

    tokenizer = nltk.load('tokenizers/punkt/{0}.pickle'.format('english'))

    for filename in corpus:
        with open(filename) as fin:
            data = fin.read()

        for start, end in tokenizer.span_tokenize(data):
            sent = data[start:end]
            sent_tokens = []
            matches = re.finditer(r'\w+|[\'\"\/^/\,\-\:\.\;\?\!\(0-9]', sent)
            for match in matches:
                mstart, mend = match.span()
                sent_tokens.append(
                    (match.group(0).lower().replace('_',''), (mstart+start, mend+start))
                    )
            yield sent_tokens


def build_trainer(tagged_sents, default_tag='DEFAULT'):
    """This builds a tagger from a corpus."""
    name_tagger = [nltk.DefaultTagger('PN').tag([name.lower() for name in names.words()])]
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
                # comment out the following line to raise to the surface all the words being tagged by this last, default tag when you run debug.py.
                (r'.*', 'NN')                     # nouns (default)
                ]

    # Right now, nothing will get to the default tagger, because the regex taggers last pattern essentially acts as a default tagger, tagging everything as NN.
    tagger0 = nltk.DefaultTagger(default_tag)
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=tagger0)
    punctuation_tagger = nltk.UnigramTagger(punctuation_tags, backoff=regexp_tagger)
    tagger1 = nltk.UnigramTagger(tagged_sents, backoff=punctuation_tagger)
    tagger2 = nltk.BigramTagger(tagged_sents, backoff=tagger1)
    tagger3 = nltk.UnigramTagger(name_tagger, backoff=tagger2)

    return tagger3


def is_verb(context):
    """This returns True if the tagged word is any form of verb, but
    it ignores the rest of the context (the second parameter)."""
    return context.current.tag.startswith('VB')


# [x] -> Int -> [[x]]
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


# FeatureContext -> dict
def get_features(context):
    """This returns the feature set for the data in the current window."""

    featureset = {
        'token0': context.current[0],
        'tag0': context.current[1],
    }
    history = reversed(list(context.history))
    for (offset, (token, tag, _start, _end)) in enumerate(history):
        featureset['token{}'.format(offset+1)] = token
        featureset['tag{}'.format(offset+1)] = tag

    return featureset


# dict -> FeatureContext -> Bool
def get_tag(_features, context, is_context=is_quote):
    """This returns the tag for the feature set to train against.
    """
    return is_context(context)


# [((TOKEN, TAG), (START, END))]
# -> [(FEATURES :: dict, SPAN :: (Int, Int), TAG :: Bool)]
def get_training_features(tagged_tokens, is_target=is_verb,
                          is_context=is_quote, feature_history=0):
    """This returns a sequence of feature sets and tags to train against for
    the input tokens."""
    window_size = feature_history + 2
    for window in windows(tagged_tokens, window_size):
        # window :: [((TOKEN, TAG), (START, END))]
        if len(window) < 2:
            continue
        # make sure that make_context and get_features can work
        # context :: FeatureContext
        context = make_context(window)
        if is_target(context):
            # features :: dict
            features = get_features(context)
            # span :: (Int, Int)
            span = (context.current.start, context.current.end)
            # tag :: Bool
            tag = get_tag(features, context, is_context)
            yield (features, span, tag)


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


# FileName -> [[((TOKEN, TAG), (START, END))]]
def get_tagged_tokens(corpus=TAGGED, testing=False):
    """This tokenizes, segments, and tags all the files in a directory."""
    if testing == True:
        # train against a smaller version of the corpus so that it doesn't take years during testing.
        tagger = build_trainer(brown.tagged_sents(categories='news'))
    else:
        tagger = build_trainer(brown.tagged_sents())
    tagged_spanned_tokens = []
    tokens_and_spans = tokenize_corpus(corpus)
    for sent in tokens_and_spans:
        to_tag = [token for (token,_) in sent]
        spans = [span for (_,span) in sent]
        sent_tagged_tokens = tagger.tag(to_tag)
        tagged_spanned_tokens.append(list(zip(sent_tagged_tokens, spans)))
    return tagged_spanned_tokens


# [[((TOKEN, TAG), (START, END))]] -> [???]
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

# this line will need to go when it's actually working.
    # tagged_tokens = [[token for (token,_) in sent]for sent in get_tagged_tokens(args.corpus)]

    # print(tagged_tokens)
    featuresets = get_all_training_features(get_tagged_tokens(args.corpus))
    featuresets = [(fs, tag) for (fs, _, tag) in featuresets]
    random.shuffle(featuresets)
    test_set, training_set = get_sets(featuresets, args.ratio)

    classifiers = [
        # nltk.ConditionalExponentialClassifier,
        # nltk.DecisionTreeClassifier,
        # nltk.MaxentClassifier,
        nltk.NaiveBayesClassifier,
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
                              args.output_dir)
            for (cls, a) in means
        )

    # TODO: MOAR TRAINING!

if __name__ == '__main__':
    main()

