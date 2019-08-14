import os
import datetime
import time
import importlib
from comet_ml import Optimizer
import data_processor
import embedding_processor
import tensorflow as tf

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()

optimiser_config = {'algorithm': 'bayes',
                    'spec': {
                        'maxCombo': 5,
                        'objective': 'minimize',
                        'metric': 'validate_loss',
                        'seed': 42,
                        'gridSize': 10,
                        'minSampleSize': 100,
                        'retryLimit': 20,
                        'retryAssignLimit': 0,
                    },
                    'parameters': {
                        'learning_rate': {'type': 'float', 'scalingType': 'uniform', 'min': 0.00002, 'max': 0.05},
                        'num_filters': {'type': 'integer', 'scalingType': 'uniform', 'min': 64, 'max': 256},
                        'kernel_size': {'type': 'integer', 'scalingType': 'uniform', 'min': 3, 'max': 10},
                        'pool_size': {'type': 'integer', 'scalingType': 'uniform', 'min': 3, 'max': 10},
                        'dropout_rate': {'type': 'float', 'scalingType': 'uniform', 'min': 0.01, 'max': 0.2},
                        'dense_units': {'type': 'integer', 'scalingType': 'uniform', 'min': 32, 'max': 256},
                    },
                    'name': 'My Bayesian Search',
                    'trials': 1,
                    }

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'cnn_opt',
                     'model_name': 'cnn',
                     'training': True,
                     'testing': True,
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

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']

# Set up comet optimiser
optimiser = Optimizer(optimiser_config)

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
for experiment in optimiser.get_experiments():

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
    # train_data = data_set.build_dataset_from_record('train', batch_size, repeat=num_epochs, is_training=True)
    train_data = data_set.build_dataset_from_record('dev', batch_size, repeat=num_epochs, is_training=True)
    eval_data = data_set.build_dataset_from_record('eval', batch_size, repeat=num_epochs, is_training=False)

    global_steps = int(len(list(train_data)))
    train_steps = int(len(list(train_data)) / num_epochs)
    eval_steps = int(len(list(eval_data)) / num_epochs)

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

    # Build the model
    print("------------------------------------")
    print("Creating model...")
    model_name = experiment_params['model_name']
    model_module = getattr(importlib.import_module('models.' + model_name.lower()), model_name.upper())
    model = model_module()

    model.build_model((max_seq_length,), len(labels), embedding_matrix, **model_params)
    print("Built model using parameters:")
    for key, value in model_params.items():
        print("{}: {}".format(key, value))

    # Create optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate=model_params['learning_rate'])

    # Display a model summary
    current_model = model.get_model()
    current_model.summary()

    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for " + str(num_epochs) + " epochs")

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

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for " + str(num_epochs) + " epochs")
