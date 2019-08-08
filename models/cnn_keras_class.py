import tensorflow as tf


class CNN(tf.keras.Model):
    """To call/create
        model = cnn_model.CNN(max_seq_length, embedding_matrix, len(labels))
        model.build(input_shape=(None, max_seq_length))

        Note: cant use model.summary() or model.to_json(). Can't export diagram

        To save weights:
        https://www.tensorflow.org/beta/guide/keras/saving_and_serializing#saving_subclassed_models
    """
    def __init__(self, max_seq_length, embedding_matrix, output_dim):
        super(CNN, self).__init__(name='CNN')
        self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                                   embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                   input_length=max_seq_length,
                                                   trainable=True,
                                                   name='embedding_layer')
        # self.embedding.build(input_shape=(1,))
        # self.embedding.set_weights([embedding_matrix])
        self.conv_1 = tf.keras.layers.Conv1D(128, 5, activation='relu')
        self.max_pool = tf.keras.layers.MaxPooling1D(5)
        self.conv_2 = tf.keras.layers.Conv1D(128, 5, activation='relu')
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs, **kwargs):
        embedding_output = self.embedding(inputs)
        output = self.conv_1(embedding_output)
        output = self.max_pool(output)
        output = self.conv_2(output)
        output = self.global_max_pool(output)
        output = self.dense_1(output)
        output = self.output_layer(output)
        return output
