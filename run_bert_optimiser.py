import os
import datetime
import time
import json
from comet_ml import Optimizer
import models
import data_processor
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tokenization import FullTokenizer

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Run Tensorflow session
sess = tf.Session()

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'bert_base_opt',
                     'model_name': 'bert',
                     'project_name': 'model-optimisation',
                     'batch_size': 32,
                     'num_epochs': 5,
                     'evaluate_steps': 500,
                     'vocab_size': 10000,
                     'max_seq_length': 128,
                     'to_tokens': False,
                     'embedding_dim': 768,
                     'embedding_type': 'bert',
                     'embedding_source': 'bert'}

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']

# Load optimiser config
optimiser_config_file = 'model_optimiser_config.json'
with open(optimiser_config_file) as json_file:
    optimiser_config = json.load(json_file)[experiment_params['model_name']]

# Set up comet optimiser
model_optimiser = Optimizer(optimiser_config)

# Data set and output paths
dataset_name = 'token_dataset' if experiment_params['to_tokens'] else 'text_dataset'
dataset_dir = os.path.join(task_name, dataset_name)
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
to_tokens = experiment_params['to_tokens']
embedding_dim = experiment_params['embedding_dim']
embedding_type = experiment_params['embedding_type']
embedding_source = experiment_params['embedding_source']

# Initialize the dataset processor
data_set = data_processor.DataProcessor(task_name, dataset_dir, max_seq_length, to_tokens=to_tokens, vocab_size=vocab_size)

# Get the BERT vocab file and casing info from the Hub module
bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

# If dataset folder is empty get the metadata and datasets to .npz files
if not os.listdir(dataset_dir):
    data_set.get_dataset()

# Load the metadata
vocabulary, labels = data_set.load_metadata()

# Loop over each experiment in the optimiser
for experiment in model_optimiser.get_experiments(project_name=experiment_params['project_name'], workspace="nathanduran", auto_output_logging='simple'):

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
    train_input_ids, train_input_masks, train_segment_ids, train_labels = data_set.build_dataset_for_bert('train', tokenizer, batch_size, is_training=True)
    # train_input_ids, train_input_masks, train_segment_ids,  train_labels = data_set.build_dataset_for_bert('dev', tokenizer, batch_size, is_training=True)
    val_input_ids, val_input_masks, val_segment_ids, val_labels = data_set.build_dataset_for_bert('val', tokenizer, batch_size, is_training=False)
    test_input_ids, test_input_masks, test_segment_ids, test_labels = data_set.build_dataset_for_bert('test', tokenizer, batch_size, is_training=False)
    global_steps = int(len(list(train_input_ids)) * num_epochs)
    train_steps = int(len(list(train_input_ids)))
    val_steps = int(len(list(val_input_ids)))
    test_steps = int(len(list(test_input_ids)))

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
    model = model_class.build_model((max_seq_length,), len(labels), [], **model_params)
    print("Built model using parameters:")
    for key, value in model_params.items():
        print("{}: {}".format(key, value))

    # Display a model summary
    model.summary()

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
                loss, accuracy = model.train_on_batch([train_input_ids[train_step], train_input_masks[train_step], train_segment_ids[train_step]], train_labels[train_step])
                train_loss.append(loss)
                train_accuracy.append(accuracy)

                experiment.log_metric('loss', np.mean(train_loss), step=global_step)
                experiment.log_metric('accuracy', np.mean(train_accuracy), step=global_step)

                # Every evaluate_steps evaluate model on validation set
                if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                    with experiment.validate():
                        for val_step in range(val_steps):

                            # Perform evaluation step on batch and record metrics
                            loss, accuracy = model.test_on_batch([val_input_ids[val_step], val_input_masks[val_step], val_segment_ids[val_step]], val_labels[val_step])
                            val_loss.append(loss)
                            val_accuracy.append(accuracy)

                            experiment.log_metric('loss', np.mean(val_loss), step=global_step)
                            experiment.log_metric('accuracy', np.mean(val_accuracy), step=global_step)

                    # Print current loss/accuracy and add to history
                    result_str = "Step: {}/{} - Train loss: {:.3f} - acc: {:.3f} - Val loss: {:.3f} - acc: {:.3f}"
                    print(result_str.format(global_step, global_steps,
                                            np.mean(train_loss), np.mean(train_accuracy) * 100,
                                            np.mean(val_loss), np.mean(val_accuracy) * 100))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")