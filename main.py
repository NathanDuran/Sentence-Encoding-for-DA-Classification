import os
from comet_ml import Experiment
import data_processor as processor
import tensorflow as tf

tf.enable_eager_execution()

# Task and experiment name
task_name = 'swda'
experiment_name = 'test'
dataset_name = 'dataset'
# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran")
# experiment = Experiment() # TODO remove this when not testing
# experiment.set_name(experiment_name)

# Data source and output paths
data_dir = task_name + '_data/'
output_dir = os.path.join(task_name + '_output', experiment_name)
dataset_dir = os.path.join(task_name + '_output', dataset_name)
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
embedding_dim = 50
max_seq_length = 128
batch_size = 1
learning_rate = 2e-5
num_epochs = 1
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
if not any(File.endswith('.tf_record') for File in os.listdir(dataset_dir)):
    data_processor.convert_all_examples()

# train_datset = data_processor.build_dataset('train', batch_size, num_epochs, is_training=True, drop_remainder=False)
dev_dataset = data_processor.build_dataset('train', batch_size, num_epochs, is_training=True, drop_remainder=False)
eval_dataset = data_processor.build_dataset('test', batch_size, num_epochs, is_training=False, drop_remainder=False)
test_dataset = data_processor.build_dataset('test', batch_size, num_epochs, is_training=False, drop_remainder=False)

# Generate the embedding matrix
embedding_matrix = data_processor.get_embedding_matrix('glove', 'glove.6B.50d', embedding_dim)

class MYModel(tf.keras.Model):
    def __init__(self):
        super(MYModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_length)
        self.embedding.build(input_shape=(1,))
        self.embedding.set_weights([embedding_matrix])

    def call(self, input, **kwargs):
        return self.embedding(input)


model = MYModel()

for batch in test_dataset.take(1):
    sentences = batch['text']
    labels = batch['label']
    # print(sentences)
    # print(labels)
    new_sent = model(sentences)
    print(new_sent)
