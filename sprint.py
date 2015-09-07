import nltk
import nltk.corpus
from nltk import word_tokenize
from nltk.corpus import wordnet as wn

from nltk.corpus import brown

f = open('corpus/1925_mrs.dalloway.txt')
raw = f.read()

# tokenizes the text
tokens = word_tokenize(raw)

# preps taggers

size = int(len(tagged_sents) * 0.9)
train_sents = tagged_sents[:size]
test_sents = tagged_sents[size:]

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

tagged_sents = brown.tagged_sents(categories='news')


# tags the tokens

tagged_tokens = t2.tag(tokens)

# sets up frequency distributions and conditional frequency distributions
fd = nltk.FreqDist(tokens)
cfd = nltk.ConditionalFreqDist(tagged_tokens)

# makes a concordance

c = nltk.ConcordanceIndex(tokens, key = lambda s: s.lower())

# working with synsets for 'said'
said_synsets = wn.synsets('said', pos='v')
said_syns = [syn.lemma_names() for syn in said_synsets]
said_syns = list(set([item for sublist in said_syns for item in sublist]))

# This goes through and pulls out all the synonyms for a particular word. "Said" is the word here. We would get a body of speech words, but does it matter that it's not necessarily pulling those from Woolf herself? Another approach that I start below is one that tries to go through and find such verbs specifically from Woolf. Should I do synonyms for those? Or just the body of verbs that she uses? Also, how to do with changes in speech?

speech_verbs = []

for verb in said_syns:
    new_syns = wn.synsets(verb, pos='v')
    new_verbs = [syn.lemma_names() for syn in new_syns]
    new_verbs = list(set([item for sublist in new_verbs for item in sublist]))
    for new_verb in new_verbs:
        speech_verbs.append(new_verb)

# pulls out the immediate contexts for quotation mark. Note - the tokenizer stores a quotation mark as two single quotes.
quote_contexts = [tokens[offset-5:offset+5] for offset in c.offsets("''")]
tagged_quote_contexts = [t2.tag(context) for context in quote_contexts]

potential_speech_tags = []
# go through and produce the verbs she uses:
for context in tagged_quote_contexts:
    context_bigrams = list(nltk.bigrams(context))
    for ((word_one, tag_one), (word_two, tag_two)) in context_bigrams:
        if tag_one in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and tag_two in ['WP', 'WP$', 'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS']:
            potential_speech_tags.append((word_one,tag_one))
            print(context)
        if tag_one in ['WP', 'WP$', 'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS'] and tag_two in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            potential_speech_tags.append((word_two, tag_two))
            print(context)

results = set([word for (word,tag) in potential_speech_tags])
