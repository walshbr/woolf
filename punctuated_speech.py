#!/usr/bin/env python
# coding=utf8


import codecs
import re


f = codecs.open('Mrs.Dalloway.txt', 'r', 'utf8')
raw_text = f.read()

clean_text = raw_text.replace('\n', '').lower()

matches = list(re.finditer(ur'“[^”]+”', clean_text))

# grabs number of punctuated phrases in novel.
number_of_punctuated_phrases = len(matches)

ordered_punctuated_phrases = [m.group() for m in matches]
raw_punctuated_phrases = "\n".join(ordered_punctuated_phrases)

# raw_punctuated phrases now has a string of all of the characters and
# punctuation marks.

# next thing to do - get rid of the punctuation marks?
print(number_of_punctuated_phrases)
print(ordered_punctuated_phrases)
print(raw_punctuated_phrases)
