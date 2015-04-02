# Read in the texts of the State of the Union addresses, using the state_union corpus reader. 
#Count occurrences of men, women, and people in each document. 
#What has happened to the usage of these words over time?

import nltk
from nltk.corpus import names

# cfd = nltk.ConditionalFreqDist(
# 	(fileid, name[-1])
# 	for fileid in names.fileids()
# 	for name in names.words(fileid))



# cfd = nltk.ConditionalFreqDist(
	
def cfd_generator():
	for fileid in names.fileids():
		for name in names.words(fileid):
			(fileid, name[-1]) 


tuples = cfd_generator()
cfd = nltk.ConditionalFreqDist(cfd_generator())

cfd.plot()