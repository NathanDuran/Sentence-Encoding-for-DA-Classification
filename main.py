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

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'cnn_test',
                     'model_name': 'cnn',
                     'training': True,
                     'testing': False,
                     'load_model': False,
                     'init_ckpt_file': 'cnn_test_ckpt-8050.h5',
                     'batch_size': 32,
                     'num_epochs': 3,
                     'evaluate_steps': 500,
                     'vocab_size': 10000,
                     'max_seq_length': 128,
                     'embedding_dim': 50,
                     'embedding_type': 'glove',
                     'embedding_source': 'glove.6B.50d'}

# Load model params
optimiser_config_file = experiment_params['model_name'] + '_params.json'
with open(os.path.join('models', optimiser_config_file)) as json_file:
    model_params = json.load(json_file)

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']
training = experiment_params['training']
testing = experiment_params['testing']
load_model = experiment_params['load_model']
init_ckpt_file = experiment_params['init_ckpt_file']

# Set up comet experiment
# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran", auto_output_logging='simple')
experiment = Experiment(auto_output_logging='simple', disabled=False)  # TODO remove this when not testing
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
# train_data = data_set.build_dataset_from_record('train', batch_size, repeat=num_epochs, is_training=True)
train_data = data_set.build_dataset_from_record('dev', batch_size, repeat=num_epochs, is_training=True)
eval_data = data_set.build_dataset_from_record('eval', batch_size, repeat=num_epochs, is_training=False)
test_data = data_set.build_dataset_from_record('test', batch_size, repeat=1, is_training=False)
global_steps = int(len(list(train_data)))
train_steps = int(len(list(train_data)) / num_epochs)
eval_steps = int(len(list(eval_data)) / num_epochs)
test_steps = int(len(list(test_data)))

print("------------------------------------")
print("Created data sets and embeddings...")
print("Maximum sequence length: " + str(max_seq_length))
print("Vocabulary size: " + str(vocab_size))
print("Embedding dimension: " + str(embedding_dim))
print("Embedding type: " + embedding_type)
print("Embedding source: " + embedding_source)
print("Global steps: " + str(global_steps))
print("Train steps: " + str(train_steps))
print("Eval steps: " + str(eval_steps))
print("Test steps: " + str(test_steps))

# Build or load the model
print("------------------------------------")
print("Creating model...")
model_name = experiment_params['model_name']
model_module = getattr(importlib.import_module('models.' + model_name.lower()), model_name.upper())
model = model_module()

# Load if checkpoint set
if load_model and os.path.exists(os.path.join(checkpoint_dir, init_ckpt_file)):
    model.load_model(os.path.join(checkpoint_dir, init_ckpt_file))
    print("Loaded model from: " + os.path.join(checkpoint_dir, init_ckpt_file))
# Else build with supplied parameters
else:
    model.build_model((max_seq_length,), len(labels), embedding_matrix, **model_params)
    print("Built model using parameters:")
    for key, value in model_params.items():
        print("{}: {}".format(key, value))

# Create optimiser
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Display a model summary and create/save a model graph definition and image
current_model = model.get_model()
current_model.summary()
model_image_file = os.path.join(output_dir, experiment_name + '_model.png')
tf.keras.utils.plot_model(current_model, to_file=model_image_file, show_shapes=True)
experiment.log_image(model_image_file)
experiment.set_model_graph(current_model.to_json())

# Train the model
if training:
    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(num_epochs) + " epochs")

    # Initialise model checkpointer
    checkpointer = checkpointer.Checkpointer(checkpoint_dir, experiment_name, model, keep_best=3, minimise=True)

    # Initialise train and evaluate metrics
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Accuracy()
    eval_loss = tf.keras.metrics.Mean()
    eval_accuracy = tf.keras.metrics.Accuracy()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}/{}".format(epoch, num_epochs))

        with experiment.train():
            for train_step, (train_text, train_labels) in enumerate(train_data.take(train_steps)):
                global_step += 1

                # Perform training step on batch and record metrics
                loss, predictions = model.training_step(optimizer, train_text, train_labels)
                train_loss(loss)
                train_accuracy(predictions, train_labels)

                experiment.log_metric('loss', train_loss.result().numpy(), step=global_step)
                experiment.log_metric('accuracy', train_accuracy.result().numpy(), step=global_step)

                # Every evaluate_steps evaluate model on evaluation set
                if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                    with experiment.validate():
                        for eval_step, (eval_text, eval_labels) in enumerate(eval_data.take(eval_steps)):

                            # Perform evaluation step on batch and record metrics
                            loss, predictions = model.evaluation_step(eval_text, eval_labels)
                            eval_loss(loss)
                            eval_accuracy(predictions, eval_labels)

                            experiment.log_metric('loss', eval_loss.result().numpy(), step=global_step)
                            experiment.log_metric('accuracy', eval_accuracy.result().numpy(), step=global_step)

                    # Print current loss/accuracy
                    result_str = "Step: {}/{} - Train loss: {:.3f} - acc: {:.3f} - Eval loss: {:.3f} - acc: {:.3f}"
                    print(result_str.format(global_step, global_steps,
                                            train_loss.result(), train_accuracy.result(),
                                            eval_loss.result(), eval_accuracy.result()))

                    # Save checkpoint if checkpointer metric improves
                    checkpointer.save_best_checkpoint(eval_loss.result(), global_step)

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")

    print("------------------------------------")
    print("Saving model...")
    checkpointer.save_checkpoint(global_step)
    experiment.log_asset_folder(checkpoint_dir)

if testing:
    # Test the model
    print("------------------------------------")
    print("Testing model...")
    start_time = time.time()
    print("Testing started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(test_steps) + " steps")

    # Initialise test metrics
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Accuracy()
    # Keep a copy of all true and predicted labels for creating evaluation metrics
    true_labels = np.empty(shape=0)
    predicted_labels = np.empty(shape=0)
    with experiment.test():
        for test_step, (test_text, test_labels) in enumerate(test_data.take(test_steps)):

            # Perform test step on batch and record metrics
            loss, predictions = model.evaluation_step(test_text, test_labels)
            test_loss(loss)
            test_accuracy(predictions, test_labels)

            experiment.log_metric('loss', test_loss.result().numpy(), step=test_step)
            experiment.log_metric('accuracy', test_accuracy.result().numpy(), step=test_step)

            # Append to lists for creating metrics
            true_labels = np.append(true_labels, test_labels.numpy().flatten())
            predicted_labels = np.append(predicted_labels, predictions.numpy().flatten())

        result_str = "Steps: {} - Test loss: {:.3f} - acc: {:.3f}"
        print(result_str.format(test_steps, test_loss.result(), test_accuracy.result()))

        # Generate metrics and confusion matrix
        metrics, metric_str = precision_recall_f1(true_labels, predicted_labels, labels)
        experiment.log_metrics(metrics)
        print(metric_str)

        confusion_matrix = plot_confusion_matrix(true_labels, predicted_labels, labels)
        confusion_matrix.show()
        confusion_matrix_file = os.path.join(output_dir, experiment_name + "_confusion_matrix.png")
        confusion_matrix.savefig(confusion_matrix_file)
        experiment.log_image(confusion_matrix_file)

        end_time = time.time()
        print("Testing took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(test_steps) + " steps")
