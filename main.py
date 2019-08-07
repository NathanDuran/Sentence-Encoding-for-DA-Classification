import math
import os
import datetime
import time
from comet_ml import Experiment
import data_processor
import embedding_processor
import tensorflow as tf

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Enable Tensorflow eager execution
tf.enable_eager_execution()

experiment_params = {'task_name': 'swda',
                     'experiment_name': 'test',
                     'training': True,
                     'testing': True,
                     'load_model': False,
                     'checkpoint_file:': 'test_best_model.h5'}

training_params = {'batch_size': 32,
                   'num_epochs': 3,
                   'evaluate_steps': 500,
                   'learning_rate': 2E-5}

dataset_params = {'vocab_size': 10000,
                  'max_seq_length': 128,
                  'embedding_dim': 50,
                  'embedding_type': 'glove',
                  'embedding_source': 'glove.6B.50d'}

# Task and experiment name
task_name = experiment_params['task_name']
experiment_name = experiment_params['experiment_name']
training = experiment_params['training']
testing = experiment_params['testing']
load_model = experiment_params['load_model']
checkpoint_file = experiment_params['checkpoint_file:']

# Set up comet experiment
# experiment = Experiment(project_name="sentence-encoding-for-da", workspace="nathanduran", auto_output_logging='simple')
experiment = Experiment(auto_output_logging='simple')  # TODO remove this when not testing
experiment.set_name(experiment_name)
# Log parameters
experiment.log_parameters(training_params)
for key, value in experiment_params.items():
    experiment.log_other(key, value)
for key, value in dataset_params.items():
    experiment.log_other(key, value)

# Data source and output paths
output_dir = os.path.join(task_name, experiment_name)
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
print("Training: " + str(training))
print("Testing: " + str(testing))

# Training parameters
batch_size = training_params['batch_size']
num_epochs = training_params['num_epochs']
evaluate_steps = training_params['evaluate_steps']  # Evaluate every this many steps
learning_rate = training_params['learning_rate']

print("------------------------------------")
print("Using parameters...")
print("Batch size: ", batch_size)
print("Epochs: ", num_epochs)
print("Evaluate every steps: ", evaluate_steps)
print("Learning rate: ", learning_rate)

# Data set parameters
vocab_size = dataset_params['vocab_size']
max_seq_length = dataset_params['max_seq_length']
embedding_dim = dataset_params['embedding_dim']
embedding_type = dataset_params['embedding_type']
embedding_source = dataset_params['embedding_source']

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
print("Maximum sequence length: ", max_seq_length)
print("Vocabulary size: ", vocab_size)
print("Embedding dimension: ", embedding_dim)
print("Embedding type: ", embedding_type)
print("Embedding source: ", embedding_source)
print("Global steps: ", global_steps)
print("Train steps: ", train_steps)
print("Eval steps: ", eval_steps)
print("Test steps: ", test_steps)

# Build or load the model
if load_model and os.path.exists(os.path.join(output_dir, checkpoint_file)):
    print("------------------------------------")
    print("Loading model...")
    model = tf.keras.models.load_model(os.path.join(output_dir, checkpoint_file))

else:
    print("------------------------------------")
    print("Building model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                  input_length=max_seq_length, trainable=True, name='embedding_layer'),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(5),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # def training_step(model_graph, x, y):
    #     with tf.GradientTape() as tape:
    #         logits = model_graph(x)
    #         loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    #
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
    #
    #     predictions = tf.argmax(logits, axis=1)
    #     return loss, predictions


model.summary()
# Create and save a model graph definition and image
model_image = os.path.join(output_dir, experiment_name + '_model.png')
tf.keras.utils.plot_model(model, to_file=model_image, show_shapes=True)
experiment.log_image(model_image)
experiment.set_model_graph(model.to_json())

# Train the model
if training:
    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Accuracy()
    eval_loss = tf.keras.metrics.Mean()
    eval_accuracy = tf.keras.metrics.Accuracy()
    global_step = 0
    best_loss = float('inf')
    best_model_ckpt = ''
    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}/{}".format(epoch, num_epochs))

        with experiment.train():
            for train_step, (train_text, train_labels) in enumerate(train_data.take(train_steps)):
                global_step += 1

                with tf.GradientTape() as tape:
                    logits = model(train_text)
                    loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=logits)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                          global_step=tf.train.get_or_create_global_step())

                train_loss(loss)
                train_accuracy(tf.argmax(logits, axis=1), train_labels)

                experiment.log_metric('loss', train_loss.result().numpy(), step=global_step)
                experiment.log_metric('accuracy', train_accuracy.result().numpy(), step=global_step)
                print("\rStep: {}/{} [{:10s}]".format(train_step + 1, train_steps,
                                                      '=' * (int(math.ceil((100 / train_steps * train_step) / 10)))), end='')

                if (train_step + 1) % evaluate_steps == 0 or (train_step + 1) == train_steps:
                    with experiment.validate():
                        for eval_step, (eval_text, eval_labels) in enumerate(eval_data.take(eval_steps)):
                            logits = model(eval_text)
                            loss = tf.losses.sparse_softmax_cross_entropy(labels=eval_labels, logits=logits)
                            eval_loss(loss)
                            eval_accuracy(tf.argmax(logits, axis=1), eval_labels)

                            experiment.log_metric('loss', eval_loss.result().numpy(), step=global_step)
                            experiment.log_metric('accuracy', eval_accuracy.result().numpy(), step=global_step)

            if eval_loss.result() < best_loss:
                if os.path.exists(best_model_ckpt):
                    os.remove(best_model_ckpt)
                best_model_ckpt = os.path.join(output_dir, experiment_name + '_model_best-{}.h5'.format(global_step))
                model.save(best_model_ckpt)
            print(" - Train loss: {:.3f} - acc: {:.3f} - Eval loss: {:.3f} - acc: {:.3f}".format(train_loss.result(),
                                                                                                 train_accuracy.result(),
                                                                                                 eval_loss.result(),
                                                                                                 eval_accuracy.result()))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")

    print("------------------------------------")
    print("Saving model...")
    final_model_ckpt = os.path.join(output_dir, experiment_name + '_model_final-{}.h5'.format(global_step))
    model.save(final_model_ckpt)
    experiment.log_asset(final_model_ckpt, overwrite=True)
    experiment.log_asset(best_model_ckpt, overwrite=True)

if testing:
    # Test the model
    print("------------------------------------")
    print("Testing model...")
    start_time = time.time()
    print("Testing started: " + datetime.datetime.now().strftime("%b %d %T") + " for", test_steps, "steps")

    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Accuracy()
    with experiment.test():
        for test_step, (test_text, test_labels) in enumerate(test_data.take(test_steps)):
            logits = model(test_text)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=test_labels, logits=logits)
            test_loss(loss)
            test_accuracy(tf.argmax(logits, axis=1), test_labels)

            experiment.log_metric('loss', test_loss.result().numpy(), step=test_step)
            experiment.log_metric('accuracy', test_accuracy.result().numpy(), step=test_step)

        print("\rStep: {}/{} [{:10s}]".format(test_step + 1, test_steps,
                                              '=' * (int(math.ceil((100 / test_steps * test_step) / 10)))), end='')

    print(" - Test loss: {:.3f} - acc: {:.3f}".format(test_loss.result(), test_accuracy.result()))
    end_time = time.time()
    print("Testing took " + str(('%.3f' % (end_time - start_time))) + " seconds for", test_steps, "steps")
