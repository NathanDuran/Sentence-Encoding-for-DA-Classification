import tensorflow as tf
import tensorflow_hub as hub


class ElmoLayer(tf.keras.layers.Layer):
    """ Wraps the Elmo module from Tensorflow Hub in a Keras Layer."""

    def __init__(self, input_mode='default', output_mode='default', **kwargs):
        """Constructor for ELMo Layer.

        Args:
            input_mode (string):
                    default = string sentence input with shape [batch_size]
                    tokens = string tokens input sequence with shape [batch_size, max_sequence_length]
            pooling (string):
                    word_emb = the character-based word representations with shape [batch_size, max_sequence_length, 512]
                    lstm_outputs1 = the first LSTM hidden state with shape [batch_size, max_sequence_length, 1024]
                    lstm_outputs2 = the second LSTM hidden state with shape [batch_size, max_sequence_length, 1024]
                    elmo = weighted sum of the 3 layers, where the weights are trainable. Tensor has shape [batch_size, max_sequence_length, 1024]
                    default = fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024]
        """
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.dimensions = 1024

        # Check input and output mode is valid
        if self.input_mode not in ["default", "tokens"]:
            raise NameError("Elmo input mode must be either default or token but is " + self.input_mode)

        self.output_modes = ["default", "word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
        if self.output_mode not in self.output_modes:
            raise NameError("Elmo output mode must be in " + str(self.output_modes) + " but is " + self.output_mode)

        super(ElmoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True, name="{}_module".format(self.name))

        if self.trainable:
            self._trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))

        super(ElmoLayer, self).build(input_shape)

    def call(self, x, mask=None):

        inputs = None
        if self.input_mode == 'default':
            inputs = tf.keras.backend.squeeze(tf.keras.backend.cast(x, tf.string), axis=1)
        # If using token input must also include the length of the sequences as a list
        elif self.input_mode == 'tokens':
            inputs = {
                "tokens": tf.keras.backend.cast(x, tf.string),
                "sequence_len": tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
            }
        result = self.elmo(inputs=inputs, as_dict=True, signature=self.input_mode)['default']
        return result

    # def compute_mask(self, inputs, mask=None):
    #     return tf.keras.backend.not_equal(inputs, '<pad>')

    def compute_output_shape(self, input_shape):
        if self.output_mode == "default":
            return input_shape[0], self.dimensions
        if self.output_mode == "word_emb":
            return input_shape[0], self.max_length, 512
        if self.output_mode == "lstm_outputs1":
            return input_shape[0], self.max_length, self.dimensions
        if self.output_mode == "lstm_outputs2":
            return input_shape[0], self.max_length, self.dimensions
        if self.output_mode == "elmo":
            return input_shape[0], self.max_length, self.dimensions

    def get_config(self):
        config = super(ElmoLayer, self).get_config()
        config['input_mode'] = self.input_mode
        config['output_mode'] = self.output_mode
        return config
