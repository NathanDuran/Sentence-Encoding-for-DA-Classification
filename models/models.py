from models import cnn, lstm, nnlm


def get_model(model_name):

    model_list = {'cnn': cnn.CNN(),
                  'lstm': lstm.LSTM(),
                  'nnlm': nnlm.NeuralNetworkLanguageModel()}

    return model_list[model_name]
