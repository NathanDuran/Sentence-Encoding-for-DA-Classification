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

# import spacy
# from spacy.tokenizer import Tokenizer
# nlp = spacy.load('en')
# tokenizer = Tokenizer(nlp.vocab)
#
# sent = "This isn't a sentence shouldn't it's."
# tokens = tokenizer(sent)
# print(tokens)
# print(len(tokens))


# import data_processor
# import os
# import tensorflow as tf
# tf.enable_eager_execution()
#
# dataset_dir = os.path.join('swda', 'dataset')
# if not os.path.exists(dataset_dir):
#     os.makedirs(dataset_dir)
# data_set = data_processor.DataProcessor('swda', dataset_dir, 128, vocab_size=10000, to_tokens=False)
# vocabulary, labels = data_set.get_metadata()
#
# test_examples = data_set.get_test_examples()
# print(test_examples[0])
#
#
# data_set.convert_examples_to_record('test', test_examples, vocabulary, labels)
# if not os.path.exists(os.path.join(dataset_dir, 'test.tf_record')):
#     print("No record!")
#
# test_data = data_set.build_dataset_from_record('test', 32, repeat=1, is_training=False)
# test_steps = int(len(list(test_data)))
# print(test_steps)
#
# for batch, (text, label) in enumerate(test_data.take(1)):
#     print(text)
#     print(label)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Import the Universal Sentence Encoder's TF Hub module
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
#
# # Compute a representation for each message, showing various lengths supported.
# word = "Elephant"
# sentence = "I am a sentence for which I would like to get its embedding."
# paragraph = (
#     "Universal Sentence Encoder embeddings also support short paragraphs. "
#     "There is no hard limit on how long the paragraph is. Roughly, the longer "
#     "the more 'diluted' the embedding will be.")
# messages = [word, sentence, paragraph]
#
# # Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)
#
# with tf.Session() as session:
#   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#   message_embeddings = session.run(embed(messages))
#
#   for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#     print("Message: {}".format(messages[i]))
#     print("Embedding size: {}".format(len(message_embedding)))
#     message_embedding_snippet = ", ".join(
#         (str(x) for x in message_embedding[:3]))
#     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

# tf.enable_eager_execution()
# sentences = ["A long sentence.", "single-word", "http://example.com"]
# sent_tensors = tf.convert_to_tensor(sentences)
#
# # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# # hub_module = hub.load(module_url, tags=[])
# # embed = hub.KerasLayer(hub_module.signatures["default"], input_shape=[], dtype=tf.string, trainable=False)
#
# module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
# embed = hub.KerasLayer(module_url, input_shape=[], dtype=tf.string, trainable=False)
#
# embeddings = embed(sentences)
# print(embeddings.shape)  # (3,128)
#
# model = tf.keras.Sequential()
# model.add(embed)
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#
# model.summary()
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x=sent_tensors, y=[1,0,1], epochs=2)

from bert_embedding import BertEmbedding
sentences = ["This is a sentences.", "This is also a sentence.", "This different."]
bert_embedding = BertEmbedding()
result = bert_embedding(sentences)

first_sentence = np.asarray(result[0])
second_sentence = np.asarray(result[1])

print(first_sentence.shape)
print(len(first_sentence[1]))
# print(first_sentence[0])

first_token_in_first_sentence = first_sentence[1][0]
first_token_in_second_sentence = second_sentence[1][0]
print(first_sentence[0][0])
print(second_sentence[0][0])
if np.array_equal(first_token_in_first_sentence, first_token_in_second_sentence):
    print("!!!Equal")
print(first_token_in_first_sentence)
print(first_token_in_second_sentence)


