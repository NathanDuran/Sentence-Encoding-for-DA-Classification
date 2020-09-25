# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- Document but remove use_crf from the run script OR update CRF to work on tf 1.15/2.X


- Test BERT large and Albert
- Add Char embeddings?
- Put embedding type in model params rather than experiment? (because different models are better with diff embeddings?)
- Update checkpointer to only keep 1 best. + add metric name and value?

# Upgrade to tf2?
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/migrate.ipynb
- Get all running from one script? OR just add run_transformers notebook
- use hugging face transformers tokeniser (+models?)
- Update dataprocessor to do BERT(transformer) tokenisation internally, so can run bert/transformers from run_transformer.py

## Models to Add
- ELMo + LSTM/Bi-LSTM
- ELMo + LSTM/Bi-LSTM + Attn
- ELMo + Attn
- BERT + LSTM

- Roberta
- GPT2
- DialoGPT
- XLNET

- https://github.com/AlexYangLi/TextClassification

## Misc
- Document the models in readme
- Document layers in readme
- Document datasets in readme

#### graph models
Language models need to be strings to_tokens=false and input_shape=(1,)
- elmo (or tokenised strings - but those suck)
- albert
- bert (input_shape can be max_seq_length)
- use
- nnlm
- mlstm_char_lm
- ConveRT (requires linux (tensorflow_text))

## To try
- Visualising attention layers
- Saving trained embeddings