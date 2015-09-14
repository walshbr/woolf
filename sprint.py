import nltk
import nltk.corpus
from nltk import word_tokenize
from nltk.corpus import names
from nltk.corpus import brown
from collections import deque
from pprint import pprint


TAGGED = 'training_passages/tagged.txt'
TEST_SET_RATIO = 0.2


def tokenize_corpus(filename):
    """Read the corpus into test and training sets."""
    with open(filename) as f:
        return word_tokenize(f.read())


def build_trainer(tagged_sents, default_tag='NN'):
    """This builds a tagger from a corpus."""
    name_tagger = [nltk.DefaultTagger('PN').tag(names.words())]

    t0 = nltk.DefaultTagger(default_tag)
    t1 = nltk.UnigramTagger(tagged_sents, backoff=t0)
    t2 = nltk.BigramTagger(tagged_sents, backoff=t1)
    t3 = nltk.UnigramTagger(name_tagger, backoff=t2)

    return t3


def is_verb(tagged_word, context):
    """This returns True if the tagged word is any form of verb, but
    it ignores the rest of the context."""
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
    (word, _) = tagged_token
    return word in {"''", "``", '"'}


def train_quotes(tagged_tokens, is_target=is_verb, is_context=is_quote,
                 amount_of_context=5):
    """This returns a conditional frequency distribution for tagged
    tokens for which is_target returns True, with the context being
    determined by is_context and the amount_of_context."""

    cfd = nltk.ConditionalFreqDist()

    for chunk in windows(tagged_tokens, 2 * amount_of_context):
        current = chunk[amount_of_context]
        if is_target(current, chunk):
            in_context = any(is_context(t) for t in chunk)
            cfd[in_context][current[0]] += 1

    return cfd


def top_probs(fd, sample=None):
    """This returns the top probabilities of the frequency
    distribution. If given `sample` is the number of items to
    return."""
    N = float(fd.N())
    return [(item, n/N) for (item, n) in fd.most_common(sample)]


def main():
    tokens = tokenize_corpus(TAGGED)
    tagger = build_trainer(brown.tagged_sents(categories='news'))
    tagged_tokens = tagger.tag(tokens)
    trained = train_quotes(tagged_tokens)
    pprint(top_probs(trained[True], 10))
    pprint(top_probs(trained[False], 10))


if __name__ == '__main__':
    main()
