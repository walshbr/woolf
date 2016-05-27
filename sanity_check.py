"""Making sure the data is acting the way I expect."""


import sys

from fset_manager import Current
from train_quotes import is_quote, is_word


def normalize(token):
    """normalize the token (lower-case, remove _)."""
    return token.lower().replace('_', '')


def main():
    """main"""
    input_file = sys.argv[1]

    with open(input_file) as fin:
        input_data = fin.read()

    manager = Current(is_quote, is_word)
    for sent in manager.tokenize_corpus(input_file):
        for (token, (start, end)) in sent:
            expected = input_data[start:end]
            if normalize(expected) != token:
                print('"{}" != "{}" ({} - {})'.format(
                    token, expected, start, end,
                    ))


if __name__ == '__main__':
    main()
