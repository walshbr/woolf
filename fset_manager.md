
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

```python

>>> from nltk.corpus import brown
>>> tagger = fset_manager.build_trainer(brown.tagged_sents())
>>> sentences = fset_manager.tag_token_spans(sentences, tagger)
>>> tagged = list(fset_manager.tag_quotes(sentences, train_quotes.is_quote))
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
>>> [sorted(f.items()) for (f, _, _) in features]
[[('!/.', True), ('matter/NN', True), ('prefatory/NN', True)], [(',/,', True), ('he/PPS', True), ('said/VBD', True)], [('./.', True), ('a/AT', True), ('entirety/NN', True), ('is/BEZ', True), ('of/IN', True), ('quote/NN', True), ('the/AT', True), ('this/DT', True)], [(',/,', True), ('said/VBD', True), ('she/PPS', True)], [('./.', True), ('a/AT', True), ('beginning/VBG', True), ('is/BEZ', True), ('quote/NN', True), ('this/DT', True)], [('./.', True), ('a/AT', True), ('is/BEZ', True), ('middle/NN', True), ('of/IN', True), ('quote/NN', True), ('the/AT', True), ('this/DT', True)], [('./.', True), ('a/AT', True), ('end/NN', True), ('is/BEZ', True), ('of/IN', True), ('quote/NN', True), ('the/AT', True), ('this/DT', True)], [('./.', True), ('expository/JJ', True), ('is/BEZ', True), ('this/DT', True), ('verbiage/NN', True)], [('!/.', True), ('finally/RB', True)]]

```

