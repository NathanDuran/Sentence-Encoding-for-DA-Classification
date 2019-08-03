
class MYModel(tf.keras.Model):
    def __init__(self):
        super(MYModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_length)
        self.embedding.build(input_shape=(1,))
        self.embedding.set_weights([embedding_matrix])

    def call(self, input, **kwargs):
        return self.embedding(input)
