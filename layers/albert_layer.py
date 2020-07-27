import tensorflow as tf
import tensorflow_hub as hub


class AlbertLayer(tf.keras.layers.Layer):
    """ Wraps the ALBERT module from Tensorflow Hub in a Keras Layer."""

    def __init__(self, output_mode="sequence", albert_model='base', **kwargs):
        """Constructor for BERT Layer.

        Args:
            output_mode (string):
                    pooled = Pooled output of the entire sequence with shape [batch_size, hidden_size]
                    sequence = Output every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size]
            albert_model (string): Determines the URL for the ALBERT module base, large etc. Default is base.
        """
        self.albert = None
        self.trainable = True
        self.output_mode = output_mode
        self.hidden_size = 768
        self.albert_model = albert_model.lower()
        self.albert_url = "https://tfhub.dev/google/albert_base/3"

        if self.output_mode not in ["pooled", "sequence"]:
            raise NameError("ALBERT output_mode must be either pool or sequence but is " + self.output_mode)

        if self.albert_model == 'base':
            self.hidden_size = 768
            self.albert_url = "https://tfhub.dev/google/albert_base/3"
        elif self.albert_model == 'large':
            self.hidden_size = 1024
            self.albert_url = "https://tfhub.dev/google/albert_large/3"
        elif self.albert_model == 'xlarge':
            self.hidden_size = 2048
            self.albert_url = "https://tfhub.dev/google/albert_xlarge/3"
        elif self.albert_model == 'xxlarge':
            self.hidden_size = 4096
            self.albert_url = "https://tfhub.dev/google/albert_xxlarge/3"
        else:
            raise NameError("Unable to determine ALBERT model type."
                            "Should be one of 'base', 'large', 'xlarge' or 'xxlarge' but was " + self.albert_model)

        super(AlbertLayer, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'albert': self.albert,
            'trainable': self.trainable,
            'hidden_size': self.hidden_size,
            'output_mode': self.output_mode,
            'albert_url': self.albert_url,
        })
        return config

    def build(self, input_shape):
        self.albert = hub.Module(self.albert_url, trainable=self.trainable, name="{}_module".format(self.name))
        super(AlbertLayer, self).build(input_shape)

    def call(self, x, mask=None):
        outputs = self.albert(x, signature='tokens', as_dict=True)
        if self.output_mode == 'pooled':
            return outputs['pooled_output']
        elif self.output_mode == 'sequence':
            return outputs['sequence_output']

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.hidden_size
