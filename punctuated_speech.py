from collections import Iterable
import string

# Pulls in the text
f = open('Mrs.Dalloway.txt', 'r')
raw_text = f.read()

# Cleans the text
clean_text = raw_text.replace('\n', '')

# Strips text into characters
characters_list = list(clean_text)

# Stores index positions of all open quotes and smart quotes
open_quote_indices = []
close_quote_indices = []
characters_dict = dict(enumerate(characters_list))

for key, value in characters_dict.items():
	if value == "“":
		open_quote_indices.append(key)
	elif value == "”":
		close_quote_indices.append(key)
	else:
		pass

#grabs number of punctuated phrases in novel. 
number_of_punctuated_phrases = len(open_quote_indices)

punctuated_characters = []
punctuated_indices = []
punctuated_phrases = []
ordered_punctuated_phrases = []


# gathers the punctuated phrases into a list and stores them in ordered_punctuated_phrases 
# which maintains their order in the novel.


i = 0
for i in range(0,number_of_punctuated_phrases):
	phrase_start = open_quote_indices[i]
	phrase_end = close_quote_indices[i]
	phrase_characters = characters_list[phrase_start:phrase_end+1]
	ordered_punctuated_phrases.append(phrase_characters)

# compresses ordered_punctuated_phrases back together. previously it was just a big list
# of individual characters.
i = 0
for item in ordered_punctuated_phrases:
	ordered_punctuated_phrases[i] = "".join(item)
	i+=1

# getting a big unordered list of the different punctuated phrase characters.	

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
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
# raw_punctuated phrases now has a string of all of the characters and punctuation marks. 

# next thing to do - get rid of the punctuation marks?
print(ordered_punctuated_phrases)	
print(raw_punctuated_phrases)
