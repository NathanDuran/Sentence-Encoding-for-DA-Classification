import tensorflow as tf
import tensorflow_hub as hub
try:
    import tensorflow_text
except ImportError:
    import platform
    import warnings
    warnings.warn("ConveRT Layer needs to import tensorflow_text but does not exist for " + platform.system())


class ConveRTLayer(tf.keras.layers.Layer):
    """ Wraps the ConveRT Tensorflow Hub module in a Keras Layer."""

    def __init__(self, **kwargs):
        self.dimensions = 512
        self.model_url = 'http://models.poly-ai.com/convert/v1/model.tar.gz'
        self.encoder = None
        super(ConveRTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.encoder = hub.Module(self.model_url, name="{}_module".format(self.name))
        super(ConveRTLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.encoder(tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions
