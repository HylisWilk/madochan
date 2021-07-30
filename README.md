# madochan
seq2seq word generator for fun

## Creating words ##

This can be done by simply calling the appropriate Class and passing a definition of a word.

```
from madochan.generator import Madochan
word_gen = Madochan()

definition = "The quality of being magnanimous but materialistic, while eating grapes."
new_word = word_gen.create_word(definition)

print(new_word)

>>>partiness
```

For a more descriptive example check out the jupyter notebook on this repo, or the colab notebook below:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HylisWilk/madochan/blob/main/examples.ipynb)
