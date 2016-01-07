"""
The class here will act as a base/interface for what takes a window
into the document and creates feature sets for training/classifying.

"""


from collections import deque, namedtuple
import os
import re

import nltk
from nltk.corpus import brown, names


TAGGED = 'training_passages/tagged_text/'

FeatureContext = namedtuple('FeatureContext',
                            ['history', 'current', 'lookahead'])
TaggedToken = namedtuple('TaggedToken', ['token', 'tag', 'start', 'end'])


def tagged_token(token_span):
    """This takes an input of ((TOKEN, TAG), (START, END)) and returns
    a TaggedToken."""
    ((token, tag), (start, end)) = token_span
    return TaggedToken(token, tag, start, end)


class AQuoteProcess:

    def make_context(self, window):
        """This makes a FeatureContext from a window of tokens (which
        will become TaggedTokens.)"""
        raise NotImplementedError()

    def get_features(self, context):
        """This returns the feature set for the data in the current window."""
        raise NotImplementedError()

    def get_tag(self, features, context):
        """This returns the tag for the feature set to train against."""
        raise NotImplementedError()

    def get_training_features(self, tagged_tokens, feature_history=0):
        """This returns a sequence of feature sets and tags to train against for
        the input tokens."""
        raise NotImplementedError()

    # [[((TOKEN, TAG), (START, END))]] -> [???]
    def get_all_training_features(self, tagged_tokens):
        """This takes tokenized, segmented, and tagged files and gets
        training features."""
        raise NotImplementedError()

    def build_trainer(self, tagged_sents, default_tag='DEFAULT'):
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

    def windows(self, seq, window_size):
        """This iterates over window_size chunks of seq."""
        window = deque()
        for item in seq:
            window.append(item)
            if len(window) > window_size:
                window.popleft()
            yield list(window)

    # FileName -> [[((TOKEN, TAG), (START, END))]]
    def get_tagged_tokens(self, corpus=TAGGED, testing=True):
        """This tokenizes, segments, and tags all the files in a directory."""
        if testing:
            # train against a smaller version of the corpus so that it
            # doesn't take years during testing.
            tagger = self.build_trainer(brown.tagged_sents(categories='news'))
        else:
            tagger = self.build_trainer(brown.tagged_sents())
        tagged_spanned_tokens = []
        tokens_and_spans = self.tokenize_corpus(corpus)
        for sent in tokens_and_spans:
            to_tag = [token for (token, _) in sent]
            spans = [span for (_, span) in sent]
            sent_tagged_tokens = tagger.tag(to_tag)
            tagged_spanned_tokens.append(list(zip(sent_tagged_tokens, spans)))
        return tagged_spanned_tokens

    # Override:
    # This needs to call ps.find_quoted_quotes to divide up each file by quotes,
    # then it can use `span_tokenizer` to identify the sentences.
    def tokenize_corpus(self, corpus):
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
                matches = re.finditer(
                    r'\w+|[\'\"\/^/\,\-\:\.\;\?\!\(0-9]', sent
                )
                for match in matches:
                    mstart, mend = match.span()
                    sent_tokens.append(
                        (match.group(0).lower().replace('_', ''),
                         (mstart+start, mend+start))
                    )
                yield sent_tokens


class QuotePoint(AQuoteProcess):
    """\
    This looks at the document as a quote-point following a token. The
    classifier is trained on the tag and token for the quote point and
    history_size preceding tokens.

    """

    def __init__(self, is_context, is_target, history_size=2):
        self.is_context = is_context
        self.is_target = is_target
        self.history_size = history_size

    def make_context(self, window):
        return FeatureContext(
            [tagged_token(t) for t in window[:-self.history_size]],
            tagged_token(window[-2]),
            tagged_token(window[-1]),
        )

    def get_features(self, context):
        featureset = {
            'token0': context.current[0],
            'tag0': context.current[1],
        }
        history = reversed(list(context.history))
        for (offset, (token, tag, _start, _end)) in enumerate(history):
            featureset['token{}'.format(offset+1)] = token
            featureset['tag{}'.format(offset+1)] = tag

        return featureset

    def get_tag(self, _features, context):
        return self.is_context(context)

    # [((TOKEN, TAG), (START, END))]
    # -> [(FEATURES :: dict, SPAN :: (Int, Int), TAG :: Bool)]
    def get_training_features(self, tagged_tokens):
        window_size = self.history_size + 2
        for window in self.windows(tagged_tokens, window_size):
            # window :: [((TOKEN, TAG), (START, END))]
            if len(window) < 2:
                continue
            # make sure that make_context and get_features can work
            # context :: FeatureContext
            context = self.make_context(window)
            if self.is_target(context):
                # features :: dict
                features = self.get_features(context)
                # span :: (Int, Int)
                span = (context.current.start, context.current.end)
                # tag :: Bool
                tag = self.get_tag(features, context)
                yield (features, span, tag)

    # [[((TOKEN, TAG), (START, END))]] -> [???]
    def get_all_training_features(self, tagged_tokens):
        """This takes tokenized, segmented, and tagged files and gets
        training features."""
        training_features = []
        for sent in tagged_tokens:
            training_features += self.get_training_features(
                sent
            )
        return training_features


class InternalStyle(QuotePoint):
    """ Assumes that we understand speech, at least in part, as a characteristic of the whole internal content of quotation marks. Rather than speech being signaled by a quotation mark and a quality of its immediately following words, it's a quality shared by all those words and marked by style in some way."""
    # So I want the feature histories to be longer…but how much longer? Start with 10. Eric do I need to relist the .is_target and whatnot here if they haven't changed? I think that I do because it will only call down or overwrite methods in full. Is that right?
    def __init__(self, is_context, is_target, history_size=5):
        QuotePoint.__init__(self, is_context, is_target, history_size)

    # def make_context(self, window):
    #     return FeatureContext(
    #         [tagged_token(t) for t in window[:-self.history_size]],
    #         tagged_token(window[-2]),
    #         tagged_token(window[-1]),
    #     )

    def get_features(self, context):
        # the lookahead is not used right now. The history is.
        featureset = {
            'token0': context.current[0],
            'tag0': context.current[1],
        }
        history = reversed(list(context.history))
        for (offset, (token, tag, _start, _end)) in enumerate(history):
            featureset['token{}'.format(offset+1)] = token
            featureset['tag{}'.format(offset+1)] = tag

        return featureset

    def get_tag(self, _features, context):
        return self.is_context(context)

    # [((TOKEN, TAG), (START, END))]
    # -> [(FEATURES :: dict, SPAN :: (Int, Int), TAG :: Bool)]
    def get_training_features(self, tagged_tokens):
        window_size = self.history_size + 2
        for window in self.windows(tagged_tokens, window_size):
            # window :: [((TOKEN, TAG), (START, END))]
            if len(window) < 2:
                continue
            # context :: FeatureContext
            context = self.make_context(window)
            if self.is_target(context):
                # features :: dict
                features = self.get_features(context)
                # span :: (Int, Int)
                span = (context.current.start, context.current.end)
                # tag :: Bool
                tag = self.get_tag(features, context)
                yield (features, span, tag)

    # [[((TOKEN, TAG), (START, END))]] -> [???]
    # def get_all_training_features(self, tagged_tokens):
    # Should all be the same.
    #     """This takes tokenized, segmented, and tagged files and gets
    #     training features."""
    #     training_features = []
    #     for sent in tagged_tokens:
    #         training_features += self.get_training_features(
    #             sent
    #         )
    #     return training_features


Current = InternalStyle
