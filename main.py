import os
import data_processor as processor
from comet_ml import Experiment
import tensorflow

# Task and experiment name
task_name = 'swda'
experiment_name = 'test'
experiment = Experiment()
experiment.set_name(experiment_name)

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = os.path.join(task_name + '_output/', experiment_name)
dataset_dir = os.path.join(output_dir, 'dataset')
# Create appropriate directories if they don't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)



print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)

# Training parameters
max_seq_length = 128
batch_size = 64
learning_rate = 2e-5
num_epochs = 3
evaluation_steps = 500  # Evaluate every this many steps

print("------------------------------------")
print("Using parameters...")
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning rate: ", learning_rate)
print("Epochs: ", num_epochs)
print("Evaluation steps: ", evaluation_steps)

# If dataset folder is empty create them...

data_processor = processor.DataProcessor(data_dir, output_dir, max_seq_length, label_type='labels')
vocabulary = data_processor.get_vocabulary()
print(vocabulary.token_to_idx)
labels = data_processor.get_labels()
print(labels.token_to_idx)
test_examples = data_processor.get_test_examples()
print(test_examples)

# nlp.data.PadSequence(128, pad_val=1, clip=True)