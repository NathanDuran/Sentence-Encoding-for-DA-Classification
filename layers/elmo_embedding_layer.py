import tensorflow as tf
import tensorflow_hub as hub


class ElmoEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.dimensions = 1024
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        if self.trainable:
            self._trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))

        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default')['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return tf.keras.backend.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions
