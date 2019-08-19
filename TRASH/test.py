# import tensorflow as tf
#
# s = tf.Session()
# print(s.list_devices())
#
# print("Test GPU available: "),
# print(tf.test.is_gpu_available())

# import pydot
# p = pydot.Dot()
# p.create()

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)

sent = "This isn't a sentence shouldn't it's."
tokens = tokenizer(sent)
print(tokens)
print(len(tokens))