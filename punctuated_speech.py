#!/usr/bin/env python
# coding=utf8


import codecs
import re


f = codecs.open('Mrs.Dalloway.txt', 'r', 'utf8')
raw_text = f.read()

clean_text = raw_text.replace('\n', '')

matches = list(re.finditer(ur'[“”]', clean_text))
open_quote_indices = [m.start() for m in matches if m.group() == u"“"]
close_quote_indices = [m.start() for m in matches if m.group() == u"”"]

# grabs number of punctuated phrases in novel.
number_of_punctuated_phrases = len(open_quote_indices)

ordered_punctuated_phrases = []

for (i, j) in zip(open_quote_indices, close_quote_indices):
    ordered_punctuated_phrases.append(clean_text[i:j+1])

raw_punctuated_phrases = "\n".join(ordered_punctuated_phrases)

# raw_punctuated phrases now has a string of all of the characters and
# punctuation marks.

# next thing to do - get rid of the punctuation marks?
print(number_of_punctuated_phrases)
print(ordered_punctuated_phrases)
print(raw_punctuated_phrases)
