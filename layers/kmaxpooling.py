import tensorflow as tf


class KMaxPooling(tf.keras.layers.Layer):
    """ Implemetation of temporal k-max pooling layer, which was first proposed in:

    Kalchbrenner, N., Grefenstette, E. and Blunsom, P. (2014) A Convolutional Neural Network for Modelling Sentences.
    Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics.

    This layer allows to detect the k most important features in a sentence, independent of their
    specific position, preserving their relative order.

    Layer code from: "https://github.com/AlexYangLi/TextClassification"
    """

    def __init__(self, k=1, **kwargs):
        self.k = k

        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into KMaxPooling muse be a 3D tensor!')
        if self.k > input_shape[1]:
            raise ValueError('detect `%d` most important features from `%d` timesteps is not allowed' %
                             (self.k, input_shape[1]))

        super(KMaxPooling, self).build(input_shape)

    def call(self, x, mask=None):
        out_shape = self.compute_output_shape(x.shape)

        permute_inputs = tf.keras.backend.permute_dimensions(x, (0, 2, 1))
        flat_permute_inputs = tf.keras.backend.reshape(permute_inputs, (-1,))
        topk_indices = tf.sort(tf.nn.top_k(permute_inputs, k=self.k)[1])

        all_indices = tf.keras.backend.reshape(tf.range(tf.keras.backend.shape(flat_permute_inputs)[0]), tf.keras.backend.shape(permute_inputs))
        to_sum_indices = tf.keras.backend.expand_dims(tf.gather(all_indices, 0, axis=-1), axis=-1)
        topk_indices += to_sum_indices

        flat_topk_indices = tf.keras.backend.reshape(topk_indices, (-1,))
        topk_output = tf.keras.backend.reshape(tf.gather(flat_permute_inputs, flat_topk_indices), tf.keras.backend.shape(topk_indices))

        y = tf.keras.backend.permute_dimensions(topk_output, (0, 2, 1))

        return tf.keras.backend.reshape(y, shape=(-1, out_shape[1], out_shape[2]))

    def compute_output_shape(self, input_shape):
        return input_shape[0].value, self.k, input_shape[-1].value

    def get_config(self):
        config = super(KMaxPooling, self).get_config()
        config['k'] = self.k
        return config
