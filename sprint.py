import nltk
import nltk.corpus
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import names
from nltk.corpus import brown

f = open('corpus/1925_mrs.dalloway.txt')
raw = f.read()

# tokenizes the text
tokens = word_tokenize(raw)

# preps training and testing corpus for taggers

tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.9)
train_sents = tagged_sents[:size]
test_sents = tagged_sents[size:]

person_names = names.words()
name_tagger = nltk.DefaultTagger('PN')
train_names = [name_tagger.tag(person_names)]
# preps taggers

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t3 = nltk.UnigramTagger(train_names, backoff=t2)

# tags the tokens

tagged_tokens = t3.tag(tokens)

# sets up frequency distributions and conditional frequency distributions
fd = nltk.FreqDist(tokens)
cfd = nltk.ConditionalFreqDist(tagged_tokens)

# makes a concordance

c = nltk.ConcordanceIndex(tokens, key = lambda s: s.lower())

def find_synonyms(verb):

    # This goes through and pulls out all the synonyms for a particular word. 

    new_syns = wn.synsets(verb, pos='v')
    new_verbs = [syn.lemma_names() for syn in new_syns]
    new_verbs = list(set([item for sublist in new_verbs for item in sublist]))
    return new_verbs

def compile_synoynms(verb_list):
    return [find_synonyms(verb) for verb in verb_list]

def find_contexts(tokens, context_marker, amount_of_context, tags=False):
    # pulls out the immediate contexts for the marker checked. Defaults to quotation mark and five tokens on either side. Also has a boolean value on the end for whether or not you want the tags to be returned as well. Note - the tokenizer stores a quotation mark as two single quotes.
    contexts = [tokens[offset-amount_of_context:offset+amount_of_context] for offset in c.offsets(context_marker)]
    if tags:
        tagged_contexts = [t2.tag(context) for context in contexts]
        return tagged_contexts
    else:
        return contexts

def find_adjacent_verbs(tokens, context_marker="''", amount_of_context=5):
    """takes in tokens, the marker around which you're looking for context, and then the number of tokens you want to look in either direction. Returns a liist of nearby verbs. Currently, it's set up to look for verbs with a noun or pronoun on either side of them."""
    potential_verbs = []
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    noun_and_pronoun_tags = ['PN', 'PPS', 'PPSS']
    # ['WP', 'WP$', 'PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS', 'PPS', 'PPSS']

    # go through and produce the verbs she uses:
    for context in find_contexts(tokens, context_marker, amount_of_context, True):
        
    # first looking for verbs and nouns immediately next to each other
        context_bigrams = list(nltk.bigrams(context))
        for ((word_one, tag_one), (word_two, tag_two)) in context_bigrams:
            if tag_one in verb_tags and tag_two in noun_and_pronoun_tags and average_similarity(word_one) > 0.1:
                potential_verbs.append((word_one,tag_one))
            if tag_one in noun_and_pronoun_tags and tag_two in verb_tags and average_similarity(word_two) > 0.1:
                potential_verbs.append((word_two, tag_two))

    # next looking for verbs with a noun two words before (accounts for some adverb situations)
        context_trigrams = list(nltk.trigrams(context))
        for ((word_one, tag_one), (word_two, tag_two), (word_three, tag_three)) in context_trigrams:
            if tag_one in noun_and_pronoun_tags and tag_three in verb_tags and average_similarity(word_three) > 0.1:
                potential_verbs.append((word_three, tag_three))

    results = set([word for (word,tag) in potential_verbs])
    return results

def average_similarity(verb):
    """Returns the average similarity between two words."""

    said_synsets = wn.synsets('said')
    new_verb_synsets = wn.synsets(verb)
    if said_synsets and new_verb_synsets:
        s = said_synsets[0].path_similarity(new_verb_synsets[0])
    return s

    # number_of_paths = 0
    # average_similarity = 0
    # total_similarity = 0
    # final_similarity = 0
    # pairs_with_averages = []

    # final average would be equal to the average of the similarities between all the senses of each word. So you need to go through and compute the similarities for each sense 
    # for said_synset in said_synsets:

    #     for new_verb_synset in new_verb_synsets:
    #         # for each sense of said, compute the average similarity with all other senses of said.
    #         similarity = said_synset.path_similarity(new_verb_synset)
    #         if not similarity:
    #             similarity = 0
    #         total_similarity += similarity
    #         number_of_paths += 1
    #     average_similarity = total_similarity / number_of_paths
    #     final_similarity += average_similarity
    # return final_similarity / len(said_synsets)

print(find_adjacent_verbs(tokens))

print(len(find_adjacent_verbs(tokens)))
# print(results)

# result_synonyms = [find_synonyms(result) for result in results]

# print("***************")
# print(result_synonyms)
# print("***************")
# print(compile_synoynms(results))


# NOTE - on synonyms - We would get a body of speech words, but does it matter that it's not necessarily pulling those from Woolf herself? Another approach that I start below is one that tries to go through and find such verbs specifically from Woolf. Should I do synonyms for those? Or just the body of verbs that she uses? Also, how to do with changes in speech?

# NOTE - LOOK AT THE PROBLEMATIC EXAMPLE OF 'CALLED' - probs using the wrong speech tags here. also difficulty with proper names.

# NOTE - synonyms seem too off the chain at present. perhaps it will work better if you can clean up the initial algorithm so it's only finding sound words - that would help a lot.

# Note - the trigrams thing doesn't seem to help a lot. it gives you different parts of speech more than anything else.

# NOTE - got the name tagger working, but it's obviously not using every name in creation. and the fictional names woolf made up don't always get pulled. should i account for that in some way?

# NOTE - the average similarity thing is a bit wonky at the moment. Before it was giving identical words a 0.2 similarity rating for each other. I think the problem is in how I'm averaging similarities together. But this seems like it could be a useful thing for culling out junk verbs that don't match our criteria. That was based on this overly complicated way of averaging the similarities between all synsets into a massive average similarity. Is just finding the similarity of the first sense enough? And what threshhold should I use for deciding when to throw words away? That's all to try and automate it. The easiest thing to do would be just to validate things. Having the threshold at 0.1 throws away 22 junk words. And we have 13 speech words.