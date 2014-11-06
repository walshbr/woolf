
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import codecs
import collections
import itertools
import re
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np


# First, we'll read in the data from the file. We'll use `codecs.open` to convert it to Unicode as we're going. That will allow us to read in and work with the curly quote characters correctly.

# In[2]:

f = codecs.open('Mrs.Dalloway.txt', 'r', 'utf8')
raw_text = f.read()
print('{} characters read.'.format(len(raw_text)))


# Now that we have the data, let's clean it up some. This will involve several steps:
# 
# 1. Get rid of newlines;
# 1. Case folding.

# In[3]:

clean_text = raw_text.replace('\n', '').lower()
print(clean_text[:75] + '...')


### Finding Quotes

# Now we can start to identify the quoted quotes. We'll use a regular expression. Let's break it down part-by-part, though, first:
# 
# * `ur` means that the code should be a `unicode` object, and it shouldn't try to escape any characters. This means that, for instance, "\n" will be interpreted as two characters (backslash and "n") not a single newline.
# * `“` looks for the first open quote.
# * `[^”]+` matches any character *except* for a close quote. The plus sign means that it needs to find at least one non-close-quote character, but it will match as many as it can find.
# * `”` finally matches the closing quote.
# 
# All put together, this regular expression should match the quoted-quotes.

# In[4]:

matches = list(re.finditer(ur'“[^”]+”', clean_text))
print('{} quoted-quotes found.'.format(len(matches)))


### Locations

# Let's see where these quotes are located. We'll essentially create a histogram of the starting locations of all of the matches. We'll process the data into [`numpy`](http://www.numpy.org/) arrays, and we'll use [`matplotlib`](http://matplotlib.org/) to draw the actual graph. (We'll closely follow the [histogram path example](http://matplotlib.org/examples/api/histogram_path_demo.html).
# 
# For a more finely grained visualization, change the value of the second parameter to `np.histogram`.

# In[8]:

locations = [m.start() for m in matches]
n, bins = np.histogram(locations, 500)

fig, ax = plt.subplots()

# corners of the rectangles
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n

XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

barpath = path.Path.make_compound_path_from_polys(XY)

patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

plt.show()


### Tokenization

# Now we have a corpus of the words. Let's break it up into tokens. We'll explicitly keep the punctuation.

# In[25]:

def take_while(pred, input_str):
    """This returns the prefix of a string that matches pred,
    and the suffix where the match stops."""
    for (i, c) in enumerate(input_str):
        if not pred(c):
            return (input_str[:i], input_str[i:])
    else:
        return (input_str, "")

def is_punct(c):
    """Since `unicode` doesn't have a punctuation predicate..."""
    return unicodedata.category(c)[0] == 'P'

def tokenize(input_str):
    """This returns an iterator over the tokens in the string."""
    rest = None

    # Since punctuations are always single characters, this isn't
    # handled by `take_while`.
    if is_punct(input_str[0]):
        yield input_str[0]
        rest = input_str[1:]

    else:
        # Try to match a string of letters or numbers. The first
        # that succeeds, yield the token and stop trying.
        for p in (unicode.isalpha, unicode.isdigit):
            token, rest = take_while(p, input_str)
            if token:
                yield token
                break
        # If it wasn't a letter or number, skip a character.
        else:
            rest = input_str[1:]

    # If there's more to try, get its tokenize and yield them.
    if rest:
        for token in tokenize(rest):
            yield token

print(list(tokenize(matches[0].group())))


### Vector Space Model

# A common way to deal with NLP documents is as a [vector space model](http://en.wikipedia.org/wiki/Vector_space_model). This takes the words out of order and just stores them as a list of frequencies. It also has to keep a look up table so you can go between words and vector indices easily.
# 
# To make this easier, let's create a class to handle the lookup tables.

# In[40]:

class VectorSpace(object):
    def __init__(self):
        self.by_index = {}
        self.by_token = {}
    def __len__(self):
        return len(self.by_index)
    def get_index(self, token):
        """If it doesn't have an index for the token, create one."""
        try:
            i = self.by_token[token]
        except KeyError:
            i = len(self.by_token)
            self.by_token[token] = i
            self.by_index[i] = token
        return i
    def lookup_token(self, i):
        """Returns None if there is no token at that position."""
        return self.by_index.get(i)
    def lookup_index(self, token):
        """Returns None if there is no index for that token."""
        return self.by_token.get(token)
    def vectorize(self, token_seq):
        """This turns a list of tokens into a numpy array."""
        v = [0] * len(self.by_token)
        for token in token_seq:
            i = self.get_index(token)
            if i < len(v):
                v[i] += 1
            elif i == len(v):
                v.append(1)
            else:
                raise Exception("Invalid index {} (len = {})".format(i, len(v)))
        return np.array(v)
    def pad(self, array):
        """This pads a numpy array to match the dimensions of this vector space."""
        padding = np.zeros(len(self) - len(array))
        return np.concatenate((array, padding))
    
vs = VectorSpace()

corpus = [list(tokenize(m.group())) for m in matches]
vs_corpus = [vs.vectorize(doc) for doc in corpus]
vs_corpus = [vs.pad(d) for d in vs_corpus]

vs_corpus[0][vs.lookup_index('the')]


### Frequencies

# We'll also want to be able to look at the frequencies of words. We'll filter out the punctuation for this, and then create a `Counter` of the fields.

# In[46]:

def frequencies(corpus):
    """This takes a list of list of tokens and returns a `Counter`."""
    return collections.Counter(
        itertools.ifilter(lambda t: not (len(t) == 1 and is_punct(t)),
                          itertools.chain.from_iterable(corpus)))

freqs = frequencies(corpus)
for (token, freq) in freqs.most_common(25):
    print(u'{}\t{}'.format(token, freq))


# In[ ]:



