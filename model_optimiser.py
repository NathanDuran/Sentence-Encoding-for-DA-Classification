import os
import datetime
import time
import importlib
import json
from comet_ml import Optimizer
from models import models
import data_processor
import embedding_processor
import optimisers
import tensorflow as tf

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()


experiment_params = {'task_name': 'swda',
                     'experiment_name': 'lstm_opt',
                     'model_name': 'lstm',
                     'project_name': 'model-optimisation',
                     'batch_size': 32,
                     'num_epochs': 5,
                     'evaluate_steps': 500,
                     'vocab_size': 10000,
                     'max_seq_length': 128,
                     'embedding_dim': 50,
                     'embedding_type': 'glove',
                     'embedding_source': 'glove.6B.50d'}

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']

# Load optimiser config
optimiser_config_file = 'optimiser_config.json'
with open(optimiser_config_file) as json_file:
    optimiser_config = json.load(json_file)[experiment_params['model_name']]

# Set up comet optimiser
model_optimiser = Optimizer(optimiser_config, project_name=experiment_params['project_name'])

# Data set and output paths
dataset_dir = os.path.join(task_name, 'dataset')
embeddings_dir = 'embeddings'

# Create appropriate directories if they don't exist
if not os.path.exists(task_name):
    os.mkdir(task_name)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)

# Training parameters
batch_size = experiment_params['batch_size']
num_epochs = experiment_params['num_epochs']
evaluate_steps = experiment_params['evaluate_steps']  # Evaluate every this many steps

print("------------------------------------")
print("Using parameters...")
print("Batch size: " + str(batch_size))
print("Epochs: " + str(num_epochs))
print("Evaluate every steps: " + str(evaluate_steps))

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

# Loop over each experiment in the optimiser
for experiment in model_optimiser.get_experiments():

    # Set up comet experiment
    experiment.set_name(experiment_name)

    # Get model params from optimiser experiment
    model_params = {}
    for key in optimiser_config['parameters'].keys():
        model_params[key] = experiment.get_parameter(key)

    # Log parameters
    experiment.log_parameters(model_params)
    for key, value in experiment_params.items():
        experiment.log_other(key, value)

    # Build tensorflow datasets from TFRecord files
    train_data = data_set.build_dataset_from_record('train', batch_size, repeat=num_epochs, is_training=True)
    # train_data = data_set.build_dataset_from_record('dev', batch_size, repeat=num_epochs, is_training=True)
    val_data = data_set.build_dataset_from_record('val', batch_size, repeat=num_epochs, is_training=False)

    global_steps = int(len(list(train_data)))
    train_steps = int(len(list(train_data)) / num_epochs)
    val_steps = int(len(list(val_data)) / num_epochs)

    print("------------------------------------")
    print("Created data sets and embeddings...")
    print("Maximum sequence length: " + str(max_seq_length))
    print("Vocabulary size: " + str(vocab_size))
    print("Embedding dimension: " + str(embedding_dim))
    print("Embedding type: " + embedding_type)
    print("Embedding source: " + embedding_source)
    print("Global steps: " + str(global_steps))
    print("Train steps: " + str(train_steps))
    print("Val steps: " + str(val_steps))

    # Build the model
    print("------------------------------------")
    print("Creating model...")
    model_class = models.get_model(experiment_params['model_name'])
    model = model_class.build_model((max_seq_length,), len(labels), embedding_matrix, **model_params)
    print("Built model using parameters:")
    for key, value in model_params.items():
        print("{}: {}".format(key, value))

    # Create optimiser
    optimiser = optimisers.get_optimiser(optimiser_type=model_params['optimiser'], lr=model_params['learning_rate'], **model_params)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

    # Display a model summary
    model.summary()

    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(num_epochs) + " epochs")

    # Initialise train and validation metrics
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.Mean()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}/{}".format(epoch, num_epochs))

        with experiment.train():
            for train_step, (train_text, train_labels) in enumerate(train_data.take(train_steps)):
                global_step += 1

                # Perform training step on batch and record metrics
                loss, accuracy = model.train_on_batch(train_text, train_labels)
                train_loss(loss)
                train_accuracy(accuracy)

                experiment.log_metric('loss', train_loss.result().numpy(), step=global_step)
                experiment.log_metric('accuracy', train_accuracy.result().numpy(), step=global_step)

                # Every evaluate_steps evaluate model on validation set
                if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                    with experiment.validate():
                        for val_step, (val_text, val_labels) in enumerate(val_data.take(val_steps)):

                            # Perform evaluation step on batch and record metrics
                            loss, accuracy = model.test_on_batch(val_text, val_labels)
                            val_loss(loss)
                            val_accuracy(accuracy)

                            experiment.log_metric('loss', val_loss.result().numpy(), step=global_step)
                            experiment.log_metric('accuracy', val_accuracy.result().numpy(), step=global_step)

                    # Print current loss/accuracy
                    result_str = "Step: {}/{} - Train loss: {:.3f} - acc: {:.3f} - Val loss: {:.3f} - acc: {:.3f}"
                    print(result_str.format(global_step, global_steps,
                                            train_loss.result(), train_accuracy.result(),
                                            val_loss.result(), val_accuracy.result()))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")
