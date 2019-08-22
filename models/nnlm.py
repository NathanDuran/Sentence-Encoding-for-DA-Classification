import tensorflow as tf
import tensorflow_hub as hub
from models.model import Model


class NeuralNetworkLanguageModel(Model):
    """Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. A Neural Probabilistic Language Model.
    Journal of Machine Learning Research, 3:1137-1155, 2003.
    """
    def __init__(self, name='UniversalSentenceEncoder'):
        super().__init__(name)
        self.name = name
        self.module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 100

        # encoder_layer = hub.KerasLayer(self.module_url, input_shape=[], dtype=tf.string, trainable=True)
        # # Define model
        # inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        # x = encoder_layer(inputs)
        # x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        # x = tf.keras.layers.Dropout(dropout_rate)(x)
        # outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)
        #
        # # Create keras model
        # model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        model = tf.keras.Sequential([
            hub.KerasLayer(self.module_url, input_shape=[], dtype=tf.string, trainable=True),
            tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')
        ])
        return model
