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
import models.keras_subclass

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()

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

# Set up comet experiment
# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran", auto_output_logging='simple')
experiment = Experiment(auto_output_logging='simple', disabled=True)  # TODO remove this when not testing
experiment.set_name(experiment_name)
# Log parameters
experiment.log_parameters(model_params)
for key, value in experiment_params.items():
    experiment.log_other(key, value)

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
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

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

# If dataset folder is empty get the metadata and datasets to TFRecords
if not os.listdir(dataset_dir):
    data_set.get_dataset()

# Load the metadata
vocabulary, labels = data_set.load_metadata()

# Generate the embedding matrix
embedding_matrix = embedding.get_embedding_matrix(embeddings_dir, embedding_source, embedding_dim, vocabulary)

# Build tensorflow datasets from TFRecord files
train_data = data_set.build_dataset_from_record('train', batch_size, repeat=num_epochs, is_training=True)
# train_data = data_set.build_dataset_from_record('dev', batch_size, repeat=num_epochs, is_training=True)
eval_data = data_set.build_dataset_from_record('eval', batch_size, repeat=num_epochs, is_training=False)
# test_data = data_set.build_dataset_from_record('test', batch_size, repeat=1, is_training=False)
global_steps = int(len(list(train_data)))
train_steps = int(len(list(train_data)) / num_epochs)
eval_steps = int(len(list(eval_data)) / num_epochs)
# test_steps = int(len(list(test_data)))


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
# OLD MODEL
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_length, weights=[embedding_matrix],
#                               mask_zero=False),
#     tf.keras.layers.LSTM(128, dropout=0.3, return_sequences=True, kernel_initializer='random_uniform',
#                          recurrent_initializer='glorot_uniform'),
#     tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, input_shape=(max_seq_length, 128))),
#     tf.keras.layers.GlobalMaxPooling1D(),
#     tf.keras.layers.Dense(len(labels), activation='softmax')
# ])

# inputs = tf.keras.Input(shape=(max_seq_length,), name='input_layer')
# x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
#                               output_dim=embedding_matrix.shape[1],  # Embedding dim
#                               embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
#                               input_length=(max_seq_length,)[0],  # Max seq length
#                               trainable=True,
#                               name='embedding_layer')(inputs)
# x = tf.keras.layers.CuDNNLSTM(128, return_sequences=True)(x)
# # x = tf.keras.layers.LSTM(lstm_units, activation='tanh',
# #                               dropout=lstm_dropout,
# #                               recurrent_dropout=recurrent_dropout,
# #                               return_sequences=True)(x)
# # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100))(x)
# x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
# x = tf.keras.layers.Dense(100, activation='relu', name='dense_1')(x)
# x = tf.keras.layers.Dropout(0.05)(x)
# outputs = tf.keras.layers.Dense(len(labels), activation='softmax', name='output_layer')(x)
# # Create keras model and set as this models parameter
# model = tf.keras.Model(inputs=inputs, outputs=outputs,)

# model = models.cnn_keras_class.CNN(max_seq_length, embedding_matrix, len(labels))
# model.build(input_shape=(None, max_seq_length))

optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, decay=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
model_image_file = os.path.join(output_dir, experiment_name + '_model.png')
tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True)
print("------------------------------------")
print("Training model...")
start_time = time.time()
print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(num_epochs) + " epochs")

# Initialise train and evaluate metrics
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.Mean()
eval_loss = tf.keras.metrics.Mean()
eval_accuracy = tf.keras.metrics.Mean()
global_step = 0
for epoch in range(1, num_epochs + 1):
    print("Epoch: {}/{}".format(epoch, num_epochs))

    with experiment.train():
        for train_step, (train_text, train_labels) in enumerate(train_data.take(train_steps)):
            global_step += 1

            # Perform training step on batch and record metrics
            loss, acc = model.train_on_batch(train_text, train_labels)

            train_loss(loss)
            train_accuracy(acc)

            # Every evaluate_steps evaluate model on evaluation set
            if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                with experiment.validate():
                    for eval_step, (eval_text, eval_labels) in enumerate(eval_data.take(eval_steps)):

                        # Perform evaluation step on batch and record metrics
                        loss, acc = model.test_on_batch(eval_text, eval_labels)
                        eval_loss(loss)
                        eval_accuracy(acc)

                # Print current loss/accuracy
                result_str = "Step: {}/{} - Train loss: {:.3f} - acc: {:.3f} - Eval loss: {:.3f} - acc: {:.3f}"
                print(result_str.format(global_step, global_steps,
                                        train_loss.result(), train_accuracy.result(),
                                        eval_loss.result(), eval_accuracy.result()))

end_time = time.time()
print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")
