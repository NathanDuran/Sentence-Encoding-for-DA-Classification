import tensorflow as tf


class Model(object):
    """Model abstract class.

    Defines the load, save and get model functionality.

    Build, training and evaluation steps are implementation specific.
    """

    def __init__(self, name='model'):
        """Constructor for Model.

        Implementation specific default parameters can be declared here.
        """
        self.name = name
        self.model = None

    def load_model(self, file_path):
        """Loads a Keras model and sets this objects model parameter.

        Args:
            file_path (str): Path/string to a .h5 keras model

        Returns:
            model (tf.keras.Model): This objects model instance
        """

        self.model = tf.keras.models.load_model(file_path)
        return self.model

    def save_model(self, file_path):
        """Saves this objects instance of a Keras model.

        Args:
            file_path (str): Path/string to save the model in .h5 format
        """

        if self.model:
            self.model.save(file_path)
        else:
            raise AttributeError("Cannot save model of type: " + str(type(self.model)) + "\n" +
                                 "Try calling load_model() or build_model() first.")

    def get_model(self):
        """Gets this objects instance of a Keras model.

        Returns:
            model (tf.keras.Model): This objects model instance
        """

        if self.model:
            return self.model
        else:
            raise AttributeError("Cannot get model of type: " + str(type(self.model)) + "\n" +
                                 "Try calling load_model() or build_model() first.")

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        """Defines the model architecture using the Keras functional API.

        Sets this objects instance of a Keras model.

        Example of 2 layer feed forward network with embeddings:

        # Unpack key word arguments
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 100

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense_1')(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model and set as this models parameter
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model = model

        Args:
            input_shape (tuple): The input shape excluding batch size, i.e (sequence_length, )
            output_shape (int): The output shape, i.e. number of classes to predict
            embedding_matrix (nb.array): A matrix of vocabulary_size rows and embedding_dim columns
            train_embeddings (bool): Whether to keep embeddings fixed during training
            **kwargs (dict): Optional dictionary of model parameters to use for specific implementations

        Returns:
            model (tf.keras.Model): This objects model instance
        """
        raise NotImplementedError()

    def training_step(self, optimizer, x, y):
        """Defines a single training step for this model, i.e. forward and backward pass on one instance/batch of data.

        Example of performing forward and backward pass, then calculating loss and predictions:

        with tf.GradientTape() as tape:
            # Forward pass and calculate loss
            logits = self.model(x, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        # Backward pass (apply gradients)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Get predictions
        predictions = tf.argmax(logits, axis=1)

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Instance of a keras optimiser
            x (Tensor): Single instance or batch of input examples
            y (Tensor): Single instance or batch of target labels

        Returns:
            loss (Tensor): Total loss for this single instance or batch of input examples
            predictions (Tensor): Predicted labels for this single instance or batch of input examples
        """
        raise NotImplementedError()

    def evaluation_step(self, x, y):
        """Defines a single evaluation step for this model, i.e. forward pass on one instance/batch of data.

        Example of performing forward pass, then calculating loss and predictions:

        # Forward pass and calculate loss
        logits = self.model(x, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        # Get predictions
        predictions = tf.argmax(logits, axis=1)

        Args:
            x (Tensor): Single instance or batch of input examples
            y (Tensor): Single instance or batch of target labels

        Returns:
            loss (Tensor): Total loss for this single instance or batch of input examples
            predictions (Tensor): Predicted labels for this single instance or batch of input examples
        """
        raise NotImplementedError()
