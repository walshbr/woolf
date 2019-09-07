# USAGE
# put this script in the same folder as your plaintext file you're trying
# to extract quoted speech from. Edit line 52 with the name of your file.
# Run it from the command line as a Python3 program like so -
# python3 find_quotations.py
# results will be stored in a results.txt file

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
        splits = re.findall(r'((?<!\w)\'.+?\'(?!\w))', text)
    else:
        splits = re.findall(r'("[^"]+")', text)
    return splits


def find_non_anglo_quotes(text):
    splits = re.findall(r'(«.+»)', text)
    return splits


def save_results(quotes):
    with open('results.txt', 'w') as fout:
        for quote in quotes:
            fout.write(quote)
            fout.write('\n\n')


def main():
    # store the path to the text you're looking at below
    filename = 'YOURTEXT.txt'
    text = read_in_text(filename)
    quotes = find_quoted_quotes(text)
    quotes.extend(find_non_anglo_quotes(text))
    save_results(quotes)


if __name__ == '__main__':
    main()
