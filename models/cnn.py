import tensorflow as tf
from models.model import Model


class CNN(Model):
    def __init__(self, name='CNN'):
        super().__init__(name)
        self.name = name
        self.model = None

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 128
        kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs.keys() else 5
        pool_size = kwargs['pool_size'] if 'pool_size' in kwargs.keys() else 5
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 100

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', name='conv_1')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size, name='max_pool')(x)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', name='conv_2')(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model and set as this models parameter
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model = model
        return model

    def training_step(self, optimizer, x, y):

        with tf.GradientTape() as tape:
            # Forward pass and calculate loss
            logits = self.model(x, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        # Backward pass (apply gradients)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Get predictions
        predictions = tf.argmax(logits, axis=1)
        return loss, predictions

    def evaluation_step(self, x, y):

        # Forward pass and calculate loss
        logits = self.model(x, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        # Get predictions
        predictions = tf.argmax(logits, axis=1)
        return loss, predictions
