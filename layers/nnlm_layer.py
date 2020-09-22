import tensorflow as tf
import tensorflow_hub as hub


class NeuralNetworkLanguageModelLayer(tf.keras.layers.Layer):
    """ Wraps the Universal Sentence Encoder module from Tensorflow Hub in a Keras Layer."""

    def __init__(self, model_url, **kwargs):
        self.dimensions = 128
        self.model_url = model_url
        self.encoder = None
        super(NeuralNetworkLanguageModelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.encoder = hub.Module(self.model_url, name="{}_module".format(self.name))
        super(NeuralNetworkLanguageModelLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.encoder(tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions
