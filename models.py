import tensorflow as tf
import tensorflow_hub as hub
import optimisers
from layers import ElmoEmbeddingLayer, BertLayer, UniversalSentenceEncoderLayer, MLSTMCharLMLayer, CRF
from layers import crf_loss, crf_accuracy


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
              'lstm_crf': LSTMCRF(),
              'lstm_attn': LSTMAttn(),
              'deep_lstm': DeepLSTM(),
              'deep_lstm_attn': DeepLSTMAttn(),
              'bi_lstm': BiLSTM(),
              'bi_lstm_attn': BiLSTMAttn(),
              'deep_bi_lstm': DeepBiLSTM(),
              'deep_bi_lstm_attn': DeepBiLSTMAttn(),
              'gru': GRU(),
              'gru_attn': GRUAttn(),
              'deep_gru': DeepGRU(),
              'deep_gru_attn': DeepGRUAttn(),
              'bi_gru': BiGRU(),
              'bi_gru_attn': BiGRUAttn(),
              'deep_bi_gru': DeepBiGRU(),
              'deep_bi_gru_attn': DeepBiGRUAttn(),
              'rcnn': RCNN(),
              'elmo': Elmo(),
              'bert': BERT(),
              'use': UniversalSentenceEncoder(),
              'nnlm': NeuralNetworkLanguageModel(),
              'mlstm_char_lm': MLSTMCharLM()}

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

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

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
    def __init__(self, name='CNN'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.002
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class CNNAttn(Model):
    def __init__(self, name='CNNAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.002
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
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
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class TextCNN(Model):
    """Implements the Text CNN model from:

    Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
    Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
    """

    def __init__(self, name='TextCNN'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.002
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class LSTM(Model):
    def __init__(self, name='LSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)
        x = lstm_layer(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class LSTMCRF(Model):
    """Uses CRF layer from keras_contrib:

    https://github.com/keras-team/keras-contrib/tree/master/keras_contrib

    Note: The labels must be of shape [batch_size, num_labels, 1].
    """

    def __init__(self, name='LSTMCRF'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0

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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)
        x = lstm_layer(x)
        outputs = CRF(output_shape, learn_mode='join', sparse_target=True)(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss=crf_loss, optimizer=optimiser, metrics=[crf_accuracy])
        return model


class LSTMAttn(Model):
    def __init__(self, name='LSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
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
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepLSTM(Model):
    def __init__(self, name='DeepLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepLSTMAttn(Model):
    def __init__(self, name='DeepLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
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
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class BiLSTM(Model):
    def __init__(self, name='BiLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)
        x = tf.keras.layers.Bidirectional(lstm_layer)(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class BiLSTMAttn(Model):
    def __init__(self, name='BiLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
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
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepBiLSTM(Model):
    def __init__(self, name='DeepBiLSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_lstm_layers = kwargs['num_lstm_layers'] if 'num_lstm_layers' in kwargs.keys() else 2
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
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
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)

        for i in range(num_lstm_layers):
            x = tf.keras.layers.Bidirectional(lstm_layer)(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepBiLSTMAttn(Model):
    def __init__(self, name='DeepBiLSTMAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        lstm_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
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
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class GRU(Model):
    def __init__(self, name='GRU'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'max'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            gru_layer = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)
        else:
            gru_layer = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                            dropout=gru_dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)
        x = gru_layer(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class GRUAttn(Model):
    def __init__(self, name='GRUAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
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
            gru_layer = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)
        else:
            gru_layer = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                            dropout=gru_dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = gru_layer(query_embedding)
        value_seq_encoding = gru_layer(value_embeddings)

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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepGRU(Model):
    def __init__(self, name='DeepGRU'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_gru_layers = kwargs['num_gru_layers'] if 'num_gru_layers' in kwargs.keys() else 2
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)

        for i in range(num_gru_layers):
            # If a GPU is available use the CUDA layer
            if tf.test.is_gpu_available() and use_gpu:
                x = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)(x)
            else:
                x = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                        dropout=gru_dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        return_sequences=True)(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepGRUAttn(Model):
    def __init__(self, name='DeepGRUAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_gru_layers = kwargs['num_gru_layers'] if 'num_gru_layers' in kwargs.keys() else 2
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # Define gru encoder model
        gru_input = tf.keras.Input(shape=(input_shape[0], embedding_matrix.shape[1]))
        # Create the first gru layer, if a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            gru_layer = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)(gru_input)
        else:
            gru_layer = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                            dropout=gru_dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            return_sequences=True)(gru_input)
        for i in range(num_gru_layers):
            if tf.test.is_gpu_available():
                gru_layer = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)(gru_layer)
            else:
                gru_layer = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                                dropout=gru_dropout,
                                                recurrent_dropout=recurrent_dropout,
                                                return_sequences=True)(gru_layer)
        gru_layers = tf.keras.Model(inputs=gru_input, outputs=gru_layer, name='gru_layers')

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = gru_layers(query_embedding)
        value_seq_encoding = gru_layers(value_embeddings)

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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class BiGRU(Model):
    def __init__(self, name='BiGRU'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'max'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            gru_layer = tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True)
        else:
            gru_layer = tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                            dropout=gru_dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)
        x = tf.keras.layers.Bidirectional(gru_layer)(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class BiGRUAttn(Model):
    def __init__(self, name='BiGRUAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
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
            gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True))
        else:
            gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                                                          dropout=gru_dropout,
                                                                          recurrent_dropout=recurrent_dropout,
                                                                          return_sequences=True))

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = gru_layer(query_embedding)
        value_seq_encoding = gru_layer(value_embeddings)

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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepBiGRU(Model):
    def __init__(self, name='DeepBiGRU'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_gru_layers = kwargs['num_gru_layers'] if 'num_gru_layers' in kwargs.keys() else 2
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        pooling = kwargs['pooling'] if 'pooling' in kwargs.keys() else 'average'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            gru_layer = tf.keras.layers.CuDNNLSTM(gru_units, return_sequences=True)
        else:
            gru_layer = tf.keras.layers.LSTM(gru_units, activation=gru_activation,
                                             dropout=gru_dropout,
                                             recurrent_dropout=recurrent_dropout,
                                             return_sequences=True)

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                      output_dim=embedding_matrix.shape[1],  # Embedding dim
                                      input_length=input_shape[0],  # Max seq length
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=train_embeddings)(inputs)

        for i in range(num_gru_layers):
            x = tf.keras.layers.Bidirectional(gru_layer)(x)

        # Define pooling type
        if pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class DeepBiGRUAttn(Model):
    def __init__(self, name='DeepBiGRUAttn'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00075
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        attention_type = kwargs['attention_type'] if 'attention_type' in kwargs.keys() else 'add'
        gru_activation = kwargs['recurrent_activation'] if 'recurrent_activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        num_gru_layers = kwargs['num_gru_layers'] if 'num_gru_layers' in kwargs.keys() else 2
        gru_units = kwargs['gru_units'] if 'gru_units' in kwargs.keys() else 256
        gru_dropout = kwargs['gru_dropout'] if 'gru_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Determine attention type
        if attention_type is 'dot':
            attention_layer = tf.keras.layers.Attention()
        else:
            attention_layer = tf.keras.layers.AdditiveAttention()

        # Define gru encoder model
        gru_input = tf.keras.Input(shape=(input_shape[0], embedding_matrix.shape[1]))
        # Create the first gru layer, if a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True))(gru_input)
        else:
            gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                                                          dropout=gru_dropout,
                                                                          recurrent_dropout=recurrent_dropout,
                                                                          return_sequences=True))(gru_input)
        for i in range(num_gru_layers):
            if tf.test.is_gpu_available():
                gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(gru_units, return_sequences=True))(gru_layer)
            else:
                gru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, activation=gru_activation,
                                                                              dropout=gru_dropout,
                                                                              recurrent_dropout=recurrent_dropout,
                                                                              return_sequences=True))(gru_layer)
        gru_layers = tf.keras.Model(inputs=gru_input, outputs=gru_layer, name='gru_layers')

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              trainable=train_embeddings)

        # Create query and value embeddings
        query_embedding = embedding(inputs)
        value_embeddings = embedding(inputs)

        # Pass through encoding layer
        query_seq_encoding = gru_layers(query_embedding)
        value_seq_encoding = gru_layers(value_embeddings)

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

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class RCNN(Model):
    """ Implements the Recurrent Convolutional Network from:

    Lai, S. et al. (2015) 'Recurrent Convolutional Neural Networks for Text Classification',
    in Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI'15).

    Model code from: "https://github.com/AlexYangLi/TextClassification"
    """

    def __init__(self, name='RCNN'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'rmsprop'
        use_gpu = kwargs['use_gpu'] if 'use_gpu' in kwargs.keys() else True
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'relu'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 256
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
        num_filters = kwargs['num_filters'] if 'num_filters' in kwargs.keys() else 64
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 128

        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],  # Vocab size
                                              output_dim=embedding_matrix.shape[1],  # Embedding dim
                                              input_length=input_shape[0],  # Max seq length
                                              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                              trainable=train_embeddings)(inputs)

        # Shift the document to the right to obtain the left-side contexts (reshape else lambda outputs 'None' shape)
        l_embedding = tf.keras.layers.Lambda(lambda x: tf.keras.backend.concatenate(
            [tf.keras.backend.zeros(shape=(tf.keras.backend.shape(x)[0], 1, tf.keras.backend.shape(x)[-1])),
             x[:, :-1]], axis=1), name='l_shift')(embedding)
        l_embedding = tf.keras.layers.Reshape((input_shape[0], embedding_matrix.shape[1]))(l_embedding)

        # Shift the document to the left to obtain the right-side contexts (reshape else lambda outputs 'None' shape)
        r_embedding = tf.keras.layers.Lambda(lambda x: tf.keras.backend.concatenate(
            [tf.keras.backend.zeros(shape=(tf.keras.backend.shape(x)[0], 1, tf.keras.backend.shape(x)[-1])),
             x[:, 1:]], axis=1), name='r_shift')(embedding)
        r_embedding = tf.keras.layers.Reshape((input_shape[0], embedding_matrix.shape[1]))(r_embedding)

        # If a GPU is available use the CUDA layer
        if tf.test.is_gpu_available() and use_gpu:
            forwards = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True, name='forwards')(l_embedding)
            backwards = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True, go_backwards=True,
                                                  name='backwards')(r_embedding)
        else:
            forwards = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                            dropout=lstm_dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            return_sequences=True,
                                            name='forwards')(l_embedding)
            backwards = tf.keras.layers.LSTM(lstm_units, activation=lstm_activation,
                                             dropout=lstm_dropout,
                                             recurrent_dropout=recurrent_dropout,
                                             return_sequences=True,
                                             go_backwards=True,
                                             name='backwards')(r_embedding)

        # Keras returns backwards LSTM outputs in reverse, so return to correct order
        backwards = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reverse(x, axes=1))(backwards)

        # Concatenate left, embedding and right contexts
        concat = tf.keras.layers.Concatenate(axis=2)([forwards, embedding, backwards])

        # Get semantic vector for each word in sequence
        y = tf.keras.layers.Conv1D(num_filters, kernel_size=1, activation="tanh")(concat)
        x = tf.keras.layers.GlobalMaxPool1D()(y)

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
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
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        embedding = ElmoEmbeddingLayer()(inputs)

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(embedding)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        # Create keras model
        model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
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
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.00002
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
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

        # Create keras model
        model = tf.keras.models.Model(inputs=bert_inputs, outputs=outputs, name=self.name)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class UniversalSentenceEncoder(Model):
    """ Uses an Universal Sentence Encoder from Tensorflow Hub as embedding layer.

    Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant,
    Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil.
    Universal Sentence Encoder. arXiv:1803.11175, 2018.

    Module url: "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    """

    def __init__(self, name='UniversalSentenceEncoder'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        embedding = UniversalSentenceEncoderLayer()(inputs)

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(embedding)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        # Create keras model
        model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class NeuralNetworkLanguageModel(Model):
    """ Uses Neural Network Language Model from Tensorflow Hub.
    
    Yoshua Bengio, Rejean Ducharme, Pascal Vincent, Christian Jauvin. A Neural Probabilistic Language Model.
    Journal of Machine Learning Research, 3:1137-1155, 2003.

    Module url: "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    """

    def __init__(self, name='NeuralNetworkLanguageModel'):
        super().__init__(name)
        self.name = name
        self.module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        # Create keras model
        model = tf.keras.Sequential([
            hub.KerasLayer(self.module_url, input_shape=[], dtype=tf.string, trainable=True),
            tf.keras.layers.Dense(dense_units, activation=dense_activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')
        ])

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model


class MLSTMCharLM(Model):
    """ Uses an mLSTM Character Language Model as embedding layer from:
    https://github.com/openai/generating-reviews-discovering-sentiment

    Radford, A., Jozefowicz, R. and Sutskever, I. (2018) 'Learning to Generate Reviews and Discovering Sentiment',
    arXiv. Available at: http://arxiv.org/abs/1704.01444

    Implements the model as described in (if return_type='mean'):
    Bothe, C. et al. (2018) 'A Context-based Approach for Dialogue Act Recognition using Simple Recurrent Neural Networks',
     in Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

    Note: batch_size and max_seq_length must be manually set for the MLSTMCharLMLayer, see mlstm_char_lm_layer.py
    """

    def __init__(self, name='mLSTMCharLM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):
        # Unpack key word arguments
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else 0.001
        optimiser = kwargs['optimiser'] if 'optimiser' in kwargs.keys() else 'adam'
        return_type = kwargs['return_type'] if 'return_type' in kwargs.keys() else 'mean'
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 32
        max_seq_length = kwargs['max_seq_length'] if 'max_seq_length' in kwargs.keys() else 640
        dense_activation = kwargs['dense_activation'] if 'dense_activation' in kwargs.keys() else 'relu'
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.02
        dense_units = kwargs['dense_units'] if 'dense_units' in kwargs.keys() else 256

        inputs = tf.keras.layers.Input(shape=input_shape, dtype="string")
        x = MLSTMCharLMLayer(batch_size=batch_size, max_seq_length=max_seq_length, return_type=return_type)(inputs)
        if return_type == 'sequence':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(dense_units, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output_layer')(x)

        # Create keras model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # Create optimiser
        optimiser = optimisers.get_optimiser(optimiser_type=optimiser, lr=learning_rate, **kwargs)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
        return model
