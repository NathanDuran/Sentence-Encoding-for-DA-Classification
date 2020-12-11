import os
import datetime
import time
import json
from comet_ml import Optimizer
import models
import data_processor
import early_stopper
import numpy as np
import tensorflow as tf

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'convert_opt',
                     'model_name': 'convert',
                     'project_name': 'sentence-encoding-for-da-model-optimisation',
                     'batch_size': 32,
                     'num_epochs': 5,
                     'evaluate_steps': 500,
                     'early_stopping': True,
                     'patience': 2,
                     'vocab_size': 10000,
                     'max_seq_length': 128,
                     'to_tokens': False,
                     'embedding_dim': 512,
                     'embedding_type': 'convert',
                     'embedding_source': 'convert'}

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']

# Load optimiser config
optimiser_config_file = 'model_optimiser_configs.json'
with open(optimiser_config_file) as json_file:
    optimiser_config = json.load(json_file)[experiment_params['model_name']]

# Set up comet optimiser
model_optimiser = Optimizer(optimiser_config)

# Data set and output paths
dataset_name = 'token_dataset' if experiment_params['to_tokens'] else 'text_dataset'
dataset_dir = os.path.join(task_name, dataset_name)
embeddings_dir = 'embeddings'

# Create appropriate directories if they don't exist
for directory in [task_name, dataset_dir, embeddings_dir]:
    if not os.path.exists(directory):
        os.mkdir(directory)

print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)

# Training parameters
batch_size = experiment_params['batch_size']
num_epochs = experiment_params['num_epochs']
evaluate_steps = experiment_params['evaluate_steps']  # Evaluate every this many steps
early_stopping = experiment_params['early_stopping']
patience = experiment_params['patience']

print("------------------------------------")
print("Using parameters...")
print("Batch size: " + str(batch_size))
print("Epochs: " + str(num_epochs))
print("Evaluate every steps: " + str(evaluate_steps))
print("Early Stopping: " + str(early_stopping))
print("Patience: " + str(patience))

# Data set parameters
vocab_size = experiment_params['vocab_size']
max_seq_length = experiment_params['max_seq_length']
to_tokens = experiment_params['to_tokens']
embedding_dim = experiment_params['embedding_dim']
embedding_type = experiment_params['embedding_type']
embedding_source = experiment_params['embedding_source']

# Initialize the dataset processor
data_set = data_processor.DataProcessor(task_name, dataset_dir, max_seq_length, vocab_size=vocab_size, to_tokens=to_tokens)

# If dataset folder is empty get the metadata and datasets
if not os.listdir(dataset_dir):
    data_set.get_dataset()

# Load the metadata
vocabulary, labels = data_set.load_metadata()

# Loop over each experiment in the optimiser
for experiment in model_optimiser.get_experiments(project_name=experiment_params['project_name'], workspace="nathanduran", auto_output_logging='simple'):

    # Run Tensorflow session
    sess = tf.Session()

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

    # Build tensorflow datasets from .npz files
    train_text, train_labels = data_set.build_dataset_from_numpy('train', batch_size, is_training=True, use_crf=False)
    val_text, val_labels = data_set.build_dataset_from_numpy('val', batch_size, is_training=False, use_crf=False)
    global_steps = int(len(list(train_text)) * num_epochs)
    train_steps = int(len(list(train_text)))
    val_steps = int(len(list(val_text)))

    print("------------------------------------")
    print("Created data sets and embeddings...")
    print("Vocabulary size: " + str(vocab_size))
    print("Maximum sequence length: " + str(max_seq_length))
    print("Using sequence tokens: " + str(to_tokens))
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
    model = model_class.build_model((1,), len(labels), [], **model_params)
    print("Built model using parameters:")
    for key, value in model_params.items():
        print("{}: {}".format(key, value))

    # Display a model summary
    model.summary()

    # Initialise early stopping monitor
    earlystopper = early_stopper.EarlyStopper(stopping=early_stopping, patience=patience, min_delta=0.01, minimise=True)

    # Initialise variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)

    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(num_epochs) + " epochs")

    # Initialise train and validation metrics
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}/{}".format(epoch, num_epochs))

        with experiment.train():
            for train_step in range(train_steps):
                global_step += 1

                # Perform training step on batch and record metrics
                loss, accuracy = model.train_on_batch(train_text[train_step], train_labels[train_step])
                train_loss.append(loss)
                train_accuracy.append(accuracy)

                experiment.log_metric('loss', np.mean(train_loss), step=global_step)
                experiment.log_metric('accuracy', np.mean(train_accuracy), step=global_step)

                # Every evaluate_steps evaluate model on validation set
                if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                    with experiment.validate():
                        for val_step in range(val_steps):

                            # Perform evaluation step on batch and record metrics
                            loss, accuracy = model.test_on_batch(val_text[val_step], val_labels[val_step])
                            val_loss.append(loss)
                            val_accuracy.append(accuracy)

                            experiment.log_metric('loss', np.mean(val_loss), step=global_step)
                            experiment.log_metric('accuracy', np.mean(val_accuracy), step=global_step)

                    # Print current loss/accuracy and add to history
                    result_str = "Step: {}/{} - Train loss: {:.3f} - acc: {:.3f} - Val loss: {:.3f} - acc: {:.3f}"
                    print(result_str.format(global_step, global_steps,
                                            np.mean(train_loss), np.mean(train_accuracy) * 100,
                                            np.mean(val_loss), np.mean(val_accuracy) * 100))

        # Check to stop training early
        if early_stopping and earlystopper.check_early_stop(float(np.mean(val_loss))):
            break

    # Close the current session
    sess.close()
    tf.keras.backend.clear_session()

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")