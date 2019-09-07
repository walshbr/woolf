# USAGE
# put this script in the same folder as your plaintext file your trying
# to extract quoted speech from.

import re


def read_in_text(filename):
    with open(filename, 'r') as fin:
        text = fin.read()
    return text


def count_quotation_marks(text):
    """"counts the number of double quotation marks in a text"""
    return len(list(re.finditer(r'"', text)))


def count_single_quotation_marks(text):
    """counts number of single quotation marks in a text"""
    return len(list(re.finditer(r"'", text)))


def find_quoted_quotes(text):
    """This returns the regex matches from finding the quoted
    quotes. Note: if the number of quotation marks is less than fifty
    it assumes that single quotes are used to designate dialogue."""
    if count_quotation_marks(text) < count_single_quotation_marks(text):
        splits = re.split(r'((?<!\w)\'.+?\'(?!\w))', text)
    else:
        splits = re.split(r'("[^"]+")', text)

    return splits


def find_non_anglo_quotes(text):
    splits = re.split(r'(«[^"]»")', text)
    return splits


def main():
    # to store the path to the text you're looking at
    filename = 'YOURTEXT.txt'
    text = read_in_text(filename)
    quotes = find_quoted_quotes(text)
    quotes.extend(find_non_anglo_quotes(text))
    print(quotes)


if __name__ == '__main__':
    main()
