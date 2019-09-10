import tensorflow as tf
import tensorflow_hub as hub


class UniversalSentenceEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.dimensions = 512
        super(UniversalSentenceEncoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.encoder = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3', name="{}_module".format(self.name))

        super(UniversalSentenceEncoderLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.encoder(tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions
