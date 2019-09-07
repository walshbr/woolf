import os
import train_quotes

"""Contains some standard debugging functions that I keep reusing. I tend to adapt them to whatever I'm currently working on. The current ones here are from when I was working on the POS tagger."""

def find_default(tagged_sentences):
    """given a list of tagged sentences, return all those words that were tagged using the default tagger."""
    nouns = []
    for sent in tagged_sentences:
        for tagged_pair in sent:
            if tagged_pair[0][1] == 'DEFAULT':
                nouns.append(tagged_pair[0][0])
    return nouns

def print_out(content, file):
    """writes the content to the file."""
    with open(file, 'w') as fout:
        fout.write('\n'.join(content))

def yield_corpus_filenames(corpus="corpus"):
    """given a function and a corpus directory, apply that function to all things in that directory"""
    if os.path.isdir(corpus):
        corpus_dir = corpus
        corpus = [
            os.path.join(corpus_dir, fn) for fn in os.listdir(corpus_dir)
        ]
    for fn in corpus:
        yield fn

def main():

    corpus = list(yield_corpus_filenames())
    sets_of_tokens = [train_quotes.get_tagged_tokens(fn) for fn in corpus]
    defaults = [find_default(text) for text in sets_of_tokens]
    defaults = set([item for sublist in defaults for item in sublist])
    print_out(defaults, "defaults.txt")

if __name__ == '__main__':
    main()


