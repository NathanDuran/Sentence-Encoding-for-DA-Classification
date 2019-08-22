import tensorflow as tf
from models.model import Model


class TextCNN(Model):
    """Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
    Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
    """

    def __init__(self, name='TextCNN'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments

        conv_activation = kwargs['conv_activation'] if 'conv_activation' in kwargs.keys() else 'elu'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 36
        kernel_sizes = [1, 2, 3, 5]
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.1
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Reshape((input_shape[0], embedding_matrix.shape[1], 1))(x)

        maxpool_pool = []
        for i in range(len(kernel_sizes)):
            conv = tf.keras.layers.Conv2D(num_filters, kernel_size=(kernel_sizes[i], embedding_matrix.shape[1]),
                                          kernel_initializer='he_normal', activation=conv_activation)(x)
            maxpool_pool.append(tf.keras.layers.MaxPool2D(pool_size=(input_shape[0] - kernel_sizes[i] + 1, 1))(conv))

        x = tf.keras.layers.Concatenate(axis=1)(maxpool_pool)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model
