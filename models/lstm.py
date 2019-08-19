import tensorflow as tf
from models.model import Model


class LSTM(Model):
    def __init__(self, name='LSTM'):
        super().__init__(name)
        self.name = name

    def build_model(self, input_shape, output_shape, embedding_matrix, train_embeddings=True, **kwargs):

        # Unpack key word arguments
        lstm_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'tanh'
        dense_activation = kwargs['activation'] if 'activation' in kwargs.keys() else 'sigmoid'
        lstm_units = kwargs['lstm_units'] if 'lstm_units' in kwargs.keys() else 128
        lstm_dropout = kwargs['lstm_dropout'] if 'lstm_dropout' in kwargs.keys() else 0.0
        recurrent_dropout = kwargs['recurrent_dropout'] if 'recurrent_dropout' in kwargs.keys() else 0.0
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
        x = tf.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)(x)
        # x = tf.keras.layers.LSTM(lstm_units, activation='tanh',
        #                               dropout=lstm_dropout,
        #                               recurrent_dropout=recurrent_dropout,
        #                               return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_units))(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name='global_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=dense_activation, name='dense_1')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='output_layer')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model
