import os
import datetime
import time
import importlib
import json
from comet_ml import Experiment
from metrics import *
import data_processor
import embedding_processor
import checkpointer
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, GlobalMaxPooling1D, Embedding
from keras.optimizers import RMSprop
from spacy.lang.en import English


# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Initialise Spacy tokeniser
tokenizer = English().Defaults.create_tokenizer(English())

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'lstm_test',
                     'model_name': 'lstm',
                     'training': True,
                     'testing': True,
                     'save_model': False,
                     'load_model': False,
                     'init_ckpt_file': '',
                     'batch_size': 32,
                     'num_epochs': 3,
                     'evaluate_steps': 500,
                     'vocab_size': 10000,
                     'max_seq_length': 128,
                     'embedding_dim': 50,
                     'embedding_type': 'glove',
                     'embedding_source': 'glove.6B.50d'}

# Load model params if file exists
optimiser_config_file = experiment_params['model_name'] + '_params.json'
if os.path.exists(os.path.join('models', optimiser_config_file)):
    with open(os.path.join('models', optimiser_config_file)) as json_file:
        model_params = json.load(json_file)
else:
    # Else use default parameters and learning rate
    model_params = {'learning_rate': 0.001}

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']
training = experiment_params['training']
testing = experiment_params['testing']
save_model = experiment_params['save_model']
load_model = experiment_params['load_model']
init_ckpt_file = experiment_params['init_ckpt_file']

# Data set and output paths
dataset_dir = os.path.join(task_name, 'dataset')
output_dir = os.path.join(task_name, experiment_name)
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
embeddings_dir = 'embeddings'

# Create appropriate directories if they don't exist
if not os.path.exists(task_name):
    os.mkdir(task_name)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)
print("Training: " + str(training))
print("Testing: " + str(testing))

# Training parameters
batch_size = experiment_params['batch_size']
num_epochs = experiment_params['num_epochs']
evaluate_steps = experiment_params['evaluate_steps']  # Evaluate every this many steps
learning_rate = model_params['learning_rate']

print("------------------------------------")
print("Using parameters...")
print("Batch size: " + str(batch_size))
print("Epochs: " + str(num_epochs))
print("Evaluate every steps: " + str(evaluate_steps))
print("Learning rate: " + str(learning_rate))

# Data set parameters
vocab_size = experiment_params['vocab_size']
max_seq_length = experiment_params['max_seq_length']
embedding_dim = experiment_params['embedding_dim']
embedding_type = experiment_params['embedding_type']
embedding_source = experiment_params['embedding_source']

# Initialize the dataset and embedding processor
data_set = data_processor.DataProcessor(task_name, dataset_dir, max_seq_length, vocab_size=vocab_size)
embedding = embedding_processor.get_embedding_processor(embedding_type)

# Load the metadata
vocabulary, labels = data_set.load_metadata()

# If dataset folder is empty get the metadata and datasets to TFRecords
train_examples = data_set.get_train_examples()

# Generate the embedding matrix
embedding_matrix = embedding.get_embedding_matrix(embeddings_dir, embedding_source, embedding_dim, vocabulary)

train_x = []
train_y = []
for example in train_examples:

    # Tokenize, convert to lowercase and remove punctuation
    tokens = tokenizer(example.text)

    tokens = [token.orth_.lower() for token in tokens]

    # Pad/truncate sequences to max_sequence_length (0 = <unk> token in vocabulary)
    tokens = [tokens[i] if i < len(tokens) else '<unk>' for i in range(max_seq_length)]

    # Convert word and label tokens to indices
    example.text = [vocabulary.token_to_idx[token] for token in tokens]
    # example.label = data_processor.to_one_hot(example.label, labels) # One hot
    example.label = labels.index(example.label)
    train_x.append(example.text)
    train_y.append(example.label)

train_x = np.array(train_x)
train_y = np.array(train_y)

print("------------------------------------")
print("Created data sets and embeddings...")
print("Maximum sequence length: " + str(max_seq_length))
print("Vocabulary size: " + str(vocab_size))
print("Embedding dimension: " + str(embedding_dim))
print("Embedding type: " + embedding_type)
print("Embedding source: " + embedding_source)

# Build the model
print("------------------------------------")
print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_length, weights=[embedding_matrix], mask_zero=False))
model.add(LSTM(128, dropout=0.3, return_sequences=True, kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform'))
model.add(TimeDistributed(Dense(128, input_shape=(max_seq_length, 128))))
model.add(GlobalMaxPooling1D())
model.add(Dense(len(labels), activation='softmax'))

optimizer = RMSprop(lr=learning_rate, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# Train the model
print("------------------------------------")
print("Training model...")

start_time = time.time()
print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=1)

end_time = time.time()
print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")


# Plot training accuracy  and loss
def plot_history(history, title='History'):
    # Create figure and title
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.suptitle(title, fontsize=14)

    # Plot accuracy
    acc = fig.add_subplot(121)
    acc.plot(history['acc'])
    acc.plot(history['val_acc'])
    acc.set_ylabel('Accuracy')
    acc.set_xlabel('Epoch')

    # Plot loss
    loss = fig.add_subplot(122)
    loss.plot(history['loss'])
    loss.plot(history['val_loss'])
    loss.set_ylabel('Loss')
    loss.set_xlabel('Epoch')
    loss.legend(['Train', 'Test'], loc='upper right')

    # Adjust layout to fit title
    fig.tight_layout()
    fig.subplots_adjust(top=0.15)

    return fig

fig = plot_history(history.history, "model_name")
fig.show()


# # Evaluate the model
# print("------------------------------------")
# print("Evaluating model...")
# model = load_model(model_dir + model_name + '.hdf5')
#
# # Test set
# test_scores = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=2)
# print("Test data: ")
# print("Loss: ", test_scores[0], " Accuracy: ", test_scores[1])
#
# # Validation set
# val_scores = model.evaluate(val_x, val_y, batch_size=batch_size, verbose=2)
# print("Validation data: ")
# print("Loss: ", val_scores[0], " Accuracy: ", val_scores[1])
#
# test_predictions = batch_prediction(model, test_data, test_x, test_y, metadata, batch_size, verbose=False)
# val_predictions = batch_prediction(model, val_data, val_x, val_y, metadata, batch_size, verbose=False)


