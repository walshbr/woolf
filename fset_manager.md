
## Initialization

```python

>>> import fset_manager
>>> import train_quotes
>>> text = """
... Prefatory matter!
... He said, "This is the entirety of a quote."
... She said, "This is beginning a quote. This is the middle of a quote. This
... is the end of a quote."
... This is expository verbiage.
... Finally!
... """
>>> len(text.splitlines())
7

```

## Break up quotes

```python

>>> quotes = fset_manager.ps.split_quoted_quotes(text)
>>> quotes
['\nPrefatory matter!\nHe said, ', '"This is the entirety of a quote."', '\nShe said, ', '"This is beginning a quote. This is the middle of a quote. This\nis the end of a quote."', '\nThis is expository verbiage.\nFinally!\n']

```

## Break up sentences

```python

>>> sentences = []
>>> for q in quotes:
...     sentences += fset_manager.split_sentences(q)
>>> sentences
['\nPrefatory matter!', '\nHe said, ', '"This is the entirety of a quote."', '\nShe said, ', '"This is beginning a quote.', 'This is the middle of a quote.', 'This\nis the end of a quote."', '\nThis is expository verbiage.', '\nFinally!\n']

```

## Tags quote states

```python

>>> fset_manager.tag_quotes(sentences, train_quotes.is_quote)
[('\nPrefatory matter!', False), ('\nHe said, ', False), ('"This is the entirety of a quote."', True), ('\nShe said, ', False), ('"This is beginning a quote.', True), ('This is the middle of a quote.', True), ('This\nis the end of a quote."', True), ('\nThis is expository verbiage.', False), ('\nFinally!\n', False)]

```

