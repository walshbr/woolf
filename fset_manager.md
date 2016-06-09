
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
>>> for s in sentences:
...     print(' '.join(token for (token, _) in s))
prefatory matter !
he said ,
" this is the entirety of a quote . "
she said ,
" this is beginning a quote .
this is the middle of a quote .
this is the end of a quote . "
this is expository verbiage .
finally !

```

## Tags quote states

**TODO**: `sentences` in this example needs to be POS-tagged.

```python

>>> pos_s = []
>>> for s in sentences:
...     pos_s.append([((token, '0'), tag) for (token, tag) in s])
>>> sentences = pos_s
>>> tagged = fset_manager.tag_quotes(sentences, train_quotes.is_quote)
>>> for (s, tag) in tagged:
...     print((' '.join(token for ((token, _), _) in s), tag))
('prefatory matter !', False)
('he said ,', False)
('" this is the entirety of a quote . "', True)
('she said ,', False)
('" this is beginning a quote .', True)
('this is the middle of a quote .', True)
('this is the end of a quote . "', True)
('this is expository verbiage .', False)
('finally !', False)

```

## Create Feature Sets from Sentences

```python

>>> manager = fset_manager.Current(train_quotes.is_quote, train_quotes.is_word)
>>> features = [manager.get_training_features(s) for (s, t) in tagged]
>>> print(features[0][0])
>>> [f for (f, _, _) in features]
[{'prefatory': 1, 'matter': 1, '!': 1}, {'he': 1, 'said': 1, ',': 1}, {'this': 1, 'is': 1, 'the': 1, 'entirety': 1, 'of': 1, 'a': 1, 'quote': 1, '.': 1}, {'she': 1, 'said': 1, ',': 1}, {'this': 1, 'is': 1, 'beginning': 1, 'a': 1, 'quote': 1, '.': 1}, {'this': 1, 'is': 1, 'the': 1, 'middle': 1, 'of': 1, 'a': 1, 'quote': 1, '.': 1}, {'this': 1, 'is': 1, 'the': 1, 'end': 1, 'of': 1, 'a': 1, 'quote': 1, '.': 1}, {'this': 1, 'is': 1, 'expsitory': 1, 'verbiage': 1, '.': 1}, {'finally': 1, '!': 1}]

```

**Before this seemed to be adding an extra tuple onto things - do we really need the tag twice? changed from [(manager.get_training_features(s), t) for (s, t) in tagged]**
**TODO**: Take POS tagging into account (needs to be passed into `tag_quotes`).
