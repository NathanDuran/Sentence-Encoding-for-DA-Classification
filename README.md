# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO

- models
- Add second dense layer to classification layers??? Time distributed?

- bi-lstm reduce num units (both recurrent and dense)?  Prob not
- bi-lstm attn increase dropout?

- GRU?
- Recurrent convolutional

- rnnlm + char level language model
- Bothe, C. et al. (2018) ‘A Context-based Approach for Dialogue Act Recognition using Simple Recurrent Neural Networks’, in Eleventh International Conference on Language Resources and Evaluation (LREC 2018). Miyazaki, Japan. Available at: http://www.informatik.uni-hamburg.de/WTM/ (Accessed: 13 August 2019).
- mLSTM https://github.com/titu1994/Keras-Multiplicative-LSTM

- elmo (try 'elmo' instead of default?) + elmo attention? More layers? 
- bert + attention? Different layers? 
- use + attention?

- Add truncating word vectors to embedding dim

- Change test to only log final values?
- Add experiment params to model? (so can compare max_seq_length, embedding_dim etc)
- Visualising attention layers
- Save train/test history + predictions?