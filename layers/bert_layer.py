import tensorflow as tf
import tensorflow_hub as hub


class BertLayer(tf.keras.layers.Layer):
    """ Wraps the BERT module from Tensorflow Hub in a Keras Layer."""

    def __init__(self, n_fine_tune_layers=12, pooling="mean_sequence",
                 bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", **kwargs):
        """Constructor for BERT Layer.

        Args:
            n_fine_tune_layers (int): Int between 1 and 12, determines how many bert layers are fine tuned
            pooling (string): Pool: pooled output of the entire sequence with shape [batch_size, hidden_size]
                           Sequence: output every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size]
                           Mean Sequence: Averaged sequence output with shape [batch_size, hidden_size]
            bert_path (string): URL to the BERT module
        """
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.hidden_size = 768
        self.pooling = pooling
        self.bert_path = bert_path

        if self.pooling not in ["pool", "sequence", "mean_sequence"]:
            raise NameError("BERT pooling type (must be either pool, sequence or mean_sequence but is" + self.pooling)

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name="{}_module".format(self.name))

        # Remove unused layers_t
        trainable_vars = self.bert.variables
        if self.pooling == "pool":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "sequence" or self.pooling == "mean_sequence":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name and not "/pooler/" in var.name]
            trainable_layers = []
        else:
            raise NameError("BERT pooling type (must be either pool, sequence or mean_sequence but is" + self.pooling)

        # Select how many layers_t to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append("encoder/layer_{}".format(str(11 - i)))

        # Update trainable vars to contain only the specified layers_t
        trainable_vars = [var for var in trainable_vars if any([l in var.name for l in trainable_layers])]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        if self.pooling == "pool":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        elif self.pooling == "sequence":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        elif self.pooling == "mean_sequence":
            sequence = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / \
                                              (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            result = masked_reduce_mean(sequence, input_mask)
        else:
            raise NameError("BERT pooling type (must be either pool, sequence or mean_sequence but is" + self.pooling)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.hidden_size

