# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?
- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)

- Test BERT large and Albert

- Upgrade to tf2 - https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/migrate.ipynb

- Collapse model configs (untie from model_name) i.e remove need for different bert_large/base configs etc
## Models to Add
- ELMo + LSTM/Bi-LSTM
- ELMo + LSTM/Bi-LSTM + Attn
- ELMo + Attn
- BERT + LSTM

- Distillbert
- GPT2!!!
 
- https://github.com/AlexYangLi/TextClassification

## Misc
- Document the models in readme
- Document layers in readme

Language models need to be strings to_tokens=false
- elmo: (or tokenised strings - but those suck)
- use:
- mlstm_char_lm:

- Document datasets in readme

#### graph models
- elmo
- bert
- use
- mlstm_char_lm
- ConveRT (requires linux (tensorflow_text))

## To try
- Visualising attention layers
- Saving trained embeddings