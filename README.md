# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO

## models
- Add second dense layer to classification layers??? Time distributed? or try CRF?

- bi-lstm attn increase dropout?

- GRU?
- Recurrent convolutional - Ribeiro, E., Ribeiro, R. and de Matos, D. M. (2018) ‘Deep Dialog Act Recognition using Multiple Token, Segment, and Context Information Representations’, arXiv. doi: arXiv:1807.08587v1.

- rnnlm + char level language model
- mLSTM https://github.com/titu1994/Keras-Multiplicative-LSTM

- elmo (try 'elmo' instead of default?) + elmo attention? More layers? 
- bert Different num layers? 
- use + attention?

## Misc
- Add truncating word vectors to embedding dim
- Try saving trained embeddings?

- Change test to only log final values?
- Add experiment params to model? (so can compare max_seq_length, embedding_dim etc)
- Visualising attention layers
- Save train/test history + predictions