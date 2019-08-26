import tensorflow as tf
import tensorflow_hub as hub
from layers import attention_with_context


def get_model(model_name):
    """Utility function for returning a Model.

    Args:
        model_name (str): The name of the Model

    Returns:
        model (tf.keras.Model): Keras model instance
    """

    models = {'cnn': CNN(),
              'text_cnn': TextCNN(),
              'lstm': LSTM(),
              'lstm_attn': LSTMAttn(),
              'deep_lstm': DeepLSTM(),
              'bi_lstm': BiLSTM(),
              'bi_lstm_attn': BiLSTMAttn(),
              'nnlm': NeuralNetworkLanguageModel()}

    if model_name.lower() not in models.keys():
        raise Exception("The given model type: '" + model_name + "' is not valid!\n" +
                        "Please select one from: " + str(list(models.keys())) + " or create one.")
    else:
        return models[model_name]


class Model(object):
    """Model abstract class."""

    def __init__(self, name='model'):
        """Constructor for Model class.

        Implementation specific default parameters can be declared here.
        """
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        """Defines the model architecture using the Keras functional API.

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

        # Create and return keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)


        Args:
            input_shape (tuple): The input shape excluding batch size, i.e (sequence_length, )
            output_shape (int): The output shape, i.e. number of classes to predict
            embedding_matrix (nb.array): A matrix of vocabulary_size rows and embedding_dim columns
            train_embeddings (bool): Whether to keep embeddings fixed during training
            **kwargs (dict): Optional dictionary of model parameters to use for specific implementations

        Returns:
            model (tf.keras.Model): Keras model instance
        """
        raise NotImplementedError()


class CNN(Model):
    def __init__(self, name='CNN_1D'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        conv_activation = kwargs['conv_activation'] if 'conv_activation' in kwargs.keys() else 'relu'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 64
        kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs.keys() else 5
        pool_size = kwargs['pool_size'] if 'pool_size' in kwargs.keys() else 8
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.27
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 225

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_1')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size, name='max_pool')(x)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_2')(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class TextCNN(Model):
    """Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
    Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
    """

    def __init__(self, name='TextCNN'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments

        conv_activation = kwargs['conv_activation'] if 'conv_activation' in kwargs.keys() else 'elu'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 36
        kernel_sizes = kwargs['kernel_sizes'] if 'kernel_sizes' in kwargs.keys() else [1, 2, 3, 5]
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.1
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Reshape((input_shape[0], embedding_matrix.shape[1], 1))(x)

        maxpool_pool = []
        for i in range(len(kernel_sizes)):
            conv = tf.keras.layers.Conv2D(num_filters, kernel_size=(kernel_sizes[i], embedding_matrix.shape[1]),
                                          kernel_initializer='he_normal', activation=conv_activation)(x)
            maxpool_pool.append(tf.keras.layers.MaxPool2D(pool_size=(input_shape[0] - kernel_sizes[i] + 1, 1))(conv))

        x = tf.keras.layers.Concatenate(axis=1)(maxpool_pool)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class LSTM(Model):
    def __init__(self, name='LSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available():
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                      dropout=lstm_dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=True)
        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = lstm_layer(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class LSTMAttn(Model):
    def __init__(self, name='LSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available():
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                      dropout=lstm_dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=True)
        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = lstm_layer(x)
        x = attention_with_context.AttentionWithContext(name='attention')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class DeepLSTM(Model):
    def __init__(self, name='DeepLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 3
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)

        # for i in range(num_lstm_layers):
        #     # If a GPU is available use the CUDA layer
        #     if tf.test.is_gpu_available():
        #         x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(x)
        #     else:
        #         x = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
        #                                           dropout=lstm_dropout,
        #                                           recurrent_dropout=recurrent_dropout,
        #                                           return_sequences=True)(x)

        x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.CuDNNLSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.CuDNNLSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(64, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class BiLSTM(Model):
    def __init__(self, name='BiLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available():
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                      dropout=lstm_dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Bidirectional(lstm_layer)(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class BiLSTMAttn(Model):
    def __init__(self, name='BiLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available():
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                      dropout=lstm_dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers.Bidirectional(lstm_layer)(x)
        x = attention_with_context.AttentionWithContext(name='attention')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class NeuralNetworkLanguageModel(Model):
    """Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. A Neural Probabilistic Language Model.
    Journal of Machine Learning Research, 3:1137-1155, 2003.
    """
    def __init__(self, name='NeuralNetworkLanguageModel'):
        super().__init__(name)
        self.name = name
        self.module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.05
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        model = tf.keras.Sequential([
            hub.KerasLayer(self.module_url, input_shape=[], dtype=tf.string, trainable=True),
            tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')
        ])
        return model