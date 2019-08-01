import os
from comet_ml import Experiment
import data_processor as processor
import tensorflow as tf

# Task and experiment name
task_name = 'swda'
experiment_name = 'test'
# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran")
# experiment = Experiment() # TODO remove this when not testing
# experiment.set_name(experiment_name)

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = os.path.join(task_name + '_output', experiment_name)
dataset_dir = os.path.join(task_name + '_output', 'dataset')
# Create appropriate directories if they don't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)

# Training parameters
vocab_size = 10000
max_seq_length = 128
batch_size = 64
learning_rate = 2e-5
num_epochs = 3
evaluation_steps = 500  # Evaluate every this many steps

print("------------------------------------")
print("Using parameters...")
print("Vocabulary size: ", vocab_size)
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning rate: ", learning_rate)
print("Epochs: ", num_epochs)
print("Evaluation steps: ", evaluation_steps)

# Initialize the dataset processor
data_processor = processor.DataProcessor(task_name, data_dir, dataset_dir, max_seq_length, vocab_size=vocab_size)

# If dataset folder is empty create the TFRecords
if not os.listdir(dataset_dir):
    data_processor.convert_all_examples()

# Generate the embedding matrix
embedding_matrix = data_processor.get_embedding_matrix('glove', 'glove.6B.50d', 50)
