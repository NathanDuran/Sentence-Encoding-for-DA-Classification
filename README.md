# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?
- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)

- Test BERT large and Albert

- Upgrade to tf2 - https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/migrate.ipynb
- Get all running from one script 
- use hugging face transformers tokeniser (+models?)

- Update dataprocessor to do BERT(transformer tokenisation internally), so can run bert from run_graph.py
## Models to Add
- ELMo + LSTM/Bi-LSTM
- ELMo + LSTM/Bi-LSTM + Attn
- ELMo + Attn
- BERT + LSTM
- Distillbert
- Roberta
- GPT2!!!
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