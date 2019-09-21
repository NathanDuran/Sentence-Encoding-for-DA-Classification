import tensorflow as tf
import tensorflow_hub as hub
from layers import ElmoEmbeddingLayer, BertLayer, UniversalSentenceEncoderLayer, MLSTMCharLMLayer


def get_model(model_name):
    """Utility function for returning a Model.

    Args:
        model_name (str): The name of the Model

    Returns:
        model (tf.keras.Model): Keras model instance
    """

    models = {'cnn': CNN(),
              'cnn_attn': CNNAttn(),
              'text_cnn': TextCNN(),
              'lstm': LSTM(),
              'lstm_attn': LSTMAttn(),
              'deep_lstm': DeepLSTM(),
              'deep_lstm_attn': DeepLSTMAttn(),
              'bi_lstm': BiLSTM(),
              'bi_lstm_attn': BiLSTMAttn(),
              'deep_bi_lstm': DeepBiLSTM(),
              'deep_bi_lstm_attn': DeepBiLSTMAttn(),
              'elmo': Elmo(),
              'bert': BERT(),
              'use': UniversalSentenceEncoder(),
              'nnlm': NeuralNetworkLanguageModel(),
              'mlstm_char_lm': MLSTMCharLM(),
              'lstm_word_lm': LSTMWordLM()}

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
        x = tf.keras.layers_t.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedding_layer')(inputs)
        x = tf.keras.layers_t.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers_t.Dense(dense_units, activation='relu', name='dense_1')(x)
        outputs = tf.keras.layers_t.Dense(output_shape, activation='softmax', name='output_layer')(x)

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
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 224

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_1')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size)(x)
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_2')(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class CNNAttn(Model):
    def __init__(self, name='CNN_1D'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        conv_activation = kwargs['conv_activation'] if 'conv_activation' in kwargs.keys() else 'relu'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 64
        kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs.keys() else 5
        pool_size = kwargs['pool_size'] if 'pool_size' in kwargs.keys() else 8
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.27
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 224

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # Define cnn encoder model
        cnn_input = tf.keras.Input(shape=(input_shape[0], embedding_matrix.shape[1]))
        cnn1 = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_1')(cnn_input)
        pool = tf.keras.layers.MaxPooling1D(pool_size)(cnn1)
        cnn_output = tf.keras.layers.Conv1D(num_filters, kernel_size, activation=conv_activation, name='conv_2')(pool)
        cnn_encoder = tf.keras.Model(inputs=cnn_input, outputs=cnn_output, name='cnn_encoder')

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              input_length=input_shape[0],  # Max seq length
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = cnn_encoder(query_embedding)
        value_seq_encoding = cnn_encoder(value_embeddings)

        # Query-value attention
        query_value_attention_seq = attention_layer([query_seq_encoding, value_seq_encoding])

        # Pool attention and encoder outputs
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        # Concatenate query and encodings
        concat = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(concat)
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
        kernel_sizes = kwargs['kernel_sizes'] if 'kernel_sizes' in kwargs.keys() else [1, 2, 3, 4, 5]
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.1
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)

        x = tf.keras.layers.Reshape((input_shape[0], embedding_matrix.shape[1], 1))(x)
        maxpool_pool = []
        for i in range(len(kernel_sizes)):
            conv = tf.keras.layers.Conv2D(num_filters, kernel_size=(kernel_sizes[i], embedding_matrix.shape[1]),
                                          kernel_initializer='he_normal', activation=conv_activation)(x)
            maxpool_pool.append(tf.keras.layers.MaxPool2D(pool_size=(input_shape[0] - kernel_sizes[i] + 1, 1))(conv))

        x = tf.keras.layers.Concatenate(axis=1)(maxpool_pool)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
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
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                              dropout=lstm_dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              return_sequences=True)
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)
        x = lstm_layer(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
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
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                              dropout=lstm_dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              input_length=input_shape[0],  # Max seq length
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = lstm_layer(query_embedding)
        value_seq_encoding = lstm_layer(value_embeddings)

        # Query-value attention
        query_value_attention_seq = attention_layer([query_seq_encoding, value_seq_encoding])

        # Pool attention and encoder outputs
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        # Concatenate query and encodings
        concat = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(concat)
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
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)

        for i in range(num_lstm_layers):
            # If a GPU is available use the CUDA layer
            if tf.test.is_gpu_available() and use_gpu:
                x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(x)
            else:
                x = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                         dropout=lstm_dropout,
                                         recurrent_dropout=recurrent_dropout,
                                         return_sequences=True)(x)

        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class DeepLSTMAttn(Model):
    def __init__(self, name='DeepLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # Define lstm encoder model
        lstm_input = tf.keras.Input(shape=(input_shape[0], embedding_matrix.shape[1]))
        # Create the first lstm layer, if a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(lstm_input)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                              dropout=lstm_dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              return_sequences=True)(lstm_input)
        for i in range(num_lstm_layers):
            if tf.test.is_gpu_available():
                lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(lstm_layer)
            else:
                lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                                  dropout=lstm_dropout,
                                                  recurrent_dropout=recurrent_dropout,
                                                  return_sequences=True)(lstm_layer)
        lstm_layers = tf.keras.Model(inputs=lstm_input, outputs=lstm_layer, name='lstm_layers')

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              input_length=input_shape[0],  # Max seq length
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = lstm_layers(query_embedding)
        value_seq_encoding = lstm_layers(value_embeddings)

        # Query-value attention
        query_value_attention_seq = attention_layer([query_seq_encoding, value_seq_encoding])

        # Pool attention and encoder outputs
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        # Concatenate query and encodings
        concat = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(concat)
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
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                              dropout=lstm_dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)
        x = tf.keras.layers.Bidirectional(lstm_layer)(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class BiLSTMAttn(Model):
    def __init__(self, name='BiLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True))
        else:
            lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                                                            dropout=lstm_dropout,
                                                                            recurrent_dropout=recurrent_dropout,
                                                                            return_sequences=True))

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              input_length=input_shape[0],  # Max seq length
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = lstm_layer(query_embedding)
        value_seq_encoding = lstm_layer(value_embeddings)

        # Query-value attention
        query_value_attention_seq = attention_layer([query_seq_encoding, value_seq_encoding])

        # Pool attention and encoder outputs
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        # Concatenate query and encodings
        concat = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(concat)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class DeepBiLSTM(Model):
    def __init__(self, name='DeepBiLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)
        else:
            lstm_layer = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                              dropout=lstm_dropout,
                                              recurrent_dropout=recurrent_dropout,
                                              return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings)(inputs)

        for i in range(num_lstm_layers):
            x = tf.keras.layers.Bidirectional(lstm_layer)(x)

        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class DeepBiLSTMAttn(Model):
    def __init__(self, name='DeepBiLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # Define lstm encoder model
        lstm_input = tf.keras.Input(shape=(input_shape[0], embedding_matrix.shape[1]))
        # Create the first lstm layer, if a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(lstm_units,
                                                                                 return_sequences=True))(lstm_input)
        else:
            lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                                                            dropout=lstm_dropout,
                                                                            recurrent_dropout=recurrent_dropout,
                                                                            return_sequences=True))(lstm_input)
        for i in range(num_lstm_layers):
            # If a GPU is available use the CUDA layer
            if tf.test.is_gpu_available():
                lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(lstm_units,
                                                                                     return_sequences=True))(lstm_layer)
            else:
                lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                                                                dropout=lstm_dropout,
                                                                                recurrent_dropout=recurrent_dropout,
                                                                                return_sequences=True))(lstm_layer)
        lstm_layers = tf.keras.Model(inputs=lstm_input, outputs=lstm_layer, name='lstm_layers')

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              input_length=input_shape[0],  # Max seq length
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = lstm_layers(query_embedding)
        value_seq_encoding = lstm_layers(value_embeddings)

        # Query-value attention
        query_value_attention_seq = attention_layer([query_seq_encoding, value_seq_encoding])

        # Pool attention and encoder outputs
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        # Concatenate query and encodings
        concat = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(concat)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


class Elmo(Model):
    """ Uses an elmo from Tensorflow Hub as embedding layer from:
    https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb

    Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer.
    Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

    Module url: "https://tfhub.dev/google/elmo/2"
    """

    def __init__(self, name='Elmo'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        embedding = ElmoEmbeddingLayer()(inputs)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(embedding)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)
        return model


class BERT(Model):
    """ Uses an BERT from Tensorflow Hub as embedding layer from:
    https://github.com/strongio/keras-bert/blob/master/keras-bert.py

    Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    arXiv preprint arXiv:1810.04805, 2018.

    Module url: "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    """

    def __init__(self, name='BERT'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        num_fine_tune_layers = kwargs['num_fine_tune_layers'] if 'num_fine_tune_layers' in kwargs.keys() else 3
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'mean_sequence'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        in_id = tf.keras.layers.Input(shape=input_shape, name="input_ids")
        in_mask = tf.keras.layers.Input(shape=input_shape, name="input_masks")
        in_segment = tf.keras.layers.Input(shape=input_shape, name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]
        bert_output = BertLayer(n_fine_tune_layers=num_fine_tune_layers, pooling=pooling)(bert_inputs)

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(bert_output)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=outputs, name=self.name)
        return model


class UniversalSentenceEncoder(Model):
    """ Uses an Universal Sentence Encoder from Tensorflow Hub as embedding layer.

    Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant,
    Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil.
    Universal Sentence Encoder. arXiv:1803.11175, 2018.

    Module url: "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    """

    def __init__(self, name='UniversalSentenceEncoder'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        embedding = UniversalSentenceEncoderLayer()(inputs)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(embedding)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)
        return model


class NeuralNetworkLanguageModel(Model):
    """ Uses Neural Network Language Model from Tensorflow Hub.
    
    Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. A Neural Probabilistic Language Model.
    Journal of Machine Learning Research, 3:1137-1155, 2003.

    Module url: "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    """

    def __init__(self, name='NeuralNetworkLanguageModel'):
        super().__init__(name)
        self.name = name
        self.module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        model = tf.keras.Sequential([
            hub.KerasLayer(self.module_url, input_shape=[], dtype=tf.string, trainable=True),
            tf.keras.layers.Dense(dense_units, activation=dense_activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')
        ])
        return model


class MLSTMCharLM(Model):
    """ Uses an mLSTM Character Language Model as embedding layer from:
    https://github.com/openai/generating-reviews-discovering-sentiment

    Radford, A., Jozefowicz, R. and Sutskever, I. (2018) ‘Learning to Generate Reviews and Discovering Sentiment’,
    arXiv. Available at: http://arxiv.org/abs/1704.01444

    Implements the model as described in:
    Bothe, C. et al. (2018) ‘A Context-based Approach for Dialogue Act Recognition using Simple Recurrent Neural Networks’,
     in Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

    Note: batch_size and max_seq_length must be manually set for the MLSTMCharLMLayer, see mlstm_char_lm_layer.py
    """

    def __init__(self, name='mLSTMCharLM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 32
        max_seq_length = kwargs['max_seq_length'] if 'max_seq_length' in kwargs.keys() else 64
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        embedding = MLSTMCharLMLayer(batch_size=batch_size, max_seq_length=max_seq_length)(inputs)  # TODO try averaging outputs?
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(embedding)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model


class LSTMWordLM(Model):
    def __init__(self, name='LSTMWordLM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        weights_file = kwargs['weights_file'] if 'weights_file' in kwargs.keys() else None
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      input_length=input_shape[0],  # Max seq length
                                      trainable=train_embeddings,
                                      name='embedd')(inputs)
        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True, name='lstm_1')(x)
            x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True, name='lstm_2')(x)
        else:
            x = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                     dropout=lstm_dropout,
                                     recurrent_dropout=recurrent_dropout,
                                     return_sequences=True,
                                     name='lstm_1')(x)
            x = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                     dropout=lstm_dropout,
                                     recurrent_dropout=recurrent_dropout,
                                     return_sequences=True,
                                     name='lstm_2')(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_units, activation=dense_activation), name='dense_1')(x)
        # x = tf.keras.layers.Dropout(dropout_rate)(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)
        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        if weights_file:
            model.load_weights(weights_file, by_name=True)
        return model