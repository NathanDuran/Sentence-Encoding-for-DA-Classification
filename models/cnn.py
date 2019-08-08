import tensorflow as tf


class CNN:
    def __init__(self, name='CNN'):
        self.name = name
        self.model = None

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
        return self.model

    def save_model(self, file_path):
        self.model.save(file_path)

    def get_model(self):
        return self.model

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True):
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Conv1D(128, 5, activation='relu', name='conv_1')(x)
        x = tf.keras.layers.MaxPooling1D(5, name='max_pool')(x)
        x = tf.keras.layers.Conv1D(128, 5, activation='relu', name='conv_2')(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model = model
        return model

    def training_step(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        predictions = tf.argmax(logits, axis=1)
        return loss, predictions

    def evaluation_step(self, x, y):

        logits = self.model(x, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        predictions = tf.argmax(logits, axis=1)
        return loss, predictions
