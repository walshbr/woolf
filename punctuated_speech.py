

import codecs
from collections import Iterable
import re


f = codecs.open('Mrs.Dalloway.txt', 'r', 'utf8')
raw_text = f.read()

clean_text = raw_text.replace('\n', '')

matches = re.findall(r'[“”]', clean_text)
open_quote_indices = [m.start() for m in matches if m.group() == "“"]
close_quote_indices = [m.start() for m in matches if m.group() == "”"]

# grabs number of punctuated phrases in novel.
number_of_punctuated_phrases = len(open_quote_indices)

ordered_punctuated_phrases = []

for (i, j) in zip(open_quote_indices, close_quote_indices):
    ordered_punctuated_phrases.append(clean_text[i:j+1])

# compresses ordered_punctuated_phrases back together. previously it was just a
# big list of individual characters.
i = 0
for item in ordered_punctuated_phrases:
        ordered_punctuated_phrases[i] = "".join(item)
        i += 1

# getting a big unordered list of the different punctuated phrase characters.

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

raw_punctuated_phrases = []


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, basestring):
            for x in flatten(item):
                yield x
        else:
            yield item


raw_punctuated_phrases = list(flatten(ordered_punctuated_phrases))
raw_punctuated_phrases = "".join(raw_punctuated_phrases)

# raw_punctuated phrases now has a string of all of the characters and
# punctuation marks.

# next thing to do - get rid of the punctuation marks?
print(ordered_punctuated_phrases)
print(raw_punctuated_phrases)
