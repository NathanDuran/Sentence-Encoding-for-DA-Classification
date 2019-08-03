import os
import inspect
from comet_ml import Experiment
import data_processor
import embedding_processor
import tensorflow as tf

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()

# Task and experiment name
task_name = 'swda'
experiment_name = 'test'

# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran")
# experiment = Experiment() # TODO remove this when not testing
# experiment.set_name(experiment_name)

# Data source and output paths
output_dir = os.path.join(task_name,  experiment_name)
dataset_dir = os.path.join(task_name, 'dataset')
embeddings_dir = 'embeddings'

# Create appropriate directories if they don't exist
if not os.path.exists(task_name):
    os.mkdir(task_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("------------------------------------")
print("Running experiment...")
print(task_name + ": " + experiment_name)

# Training parameters
vocab_size = 10000
embedding_dim = 50
embedding_type = 'glove'
embedding_source = 'glove.6B.50d'
max_seq_length = 128
batch_size = 1
learning_rate = 2e-5
num_epochs = 1
evaluation_steps = 500  # Evaluate every this many steps

print("------------------------------------")
print("Using parameters...")
print("Vocabulary size: ", vocab_size)
print("Embedding type: ", embedding_type)
print("Embedding source: ", embedding_source)
print("Maximum sequence length: ", max_seq_length)
print("Batch size: ", batch_size)
print("Learning rate: ", learning_rate)
print("Epochs: ", num_epochs)
print("Evaluation steps: ", evaluation_steps)

# Initialize the dataset and embedding processor
data_set = data_processor.DataProcessor(task_name, dataset_dir, max_seq_length, vocab_size=vocab_size)
embedding = embedding_processor.get_embedding_processor(embedding_type)

# If dataset folder is empty get the metadata and datasets to TFRecords
if not os.listdir(dataset_dir):
    data_set.get_dataset()

# Load the metadata
vocabulary, labels = data_set.load_metadata()

# Build tensorflow datasets from TFRecord files
# train_data = data_set.build_dataset_from_record('train', len(labels), batch_size, repeat=num_epochs, is_training=True)
train_data = data_set.build_dataset_from_record('dev', len(labels), batch_size, repeat=num_epochs, is_training=True)
eval_data = data_set.build_dataset_from_record('eval', len(labels), batch_size, repeat=num_epochs, is_training=False)
test_data = data_set.build_dataset_from_record('test', len(labels), batch_size, repeat=num_epochs, is_training=False)


# Generate the embedding matrix
embedding_matrix = embedding.get_embedding_matrix(embeddings_dir, embedding_source, embedding_dim, vocabulary)
print(embedding_matrix)


# # model = MYModel()
# embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), input_length=max_seq_length, trainable=True)
#
#
# model = tf.keras.Sequential([
#     embedding_layer
# ])
#
# for batch, (sentences, labels) in enumerate(test_data.take(1)):
#     print(type(batch))
#     print(batch)
#     print(sentences)
#     print(labels)
#     new_sent = model(sentences)
#     print(new_sent)
