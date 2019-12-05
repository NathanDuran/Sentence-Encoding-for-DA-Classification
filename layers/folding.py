import tensorflow as tf


class Folding(tf.keras.layers.Layer):
    """ Folding Layer code from: "https://github.com/AlexYangLi/TextClassification" """

    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into Folding Layer must be a 3D tensor!')
        super(Folding, self).build(input_shape)

    def call(self, x, mask=None):
        # Split the tensor along dimension 2 into dimension_axis_size/2 which will give us 2 tensors.
        # will raise ValueError if K.int_shape(inputs) is odd
        splits = tf.split(x, int(tf.keras.backend.int_shape(x)[-1] / 2), axis=-1)

        # Reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=-1) for split in splits]

        # Stack them up along the same axis we have reduced
        row_reduced = tf.keras.backend.stack(reduce_sums, axis=-1)
        return row_reduced

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / 2)
