# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?
- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)
https://github.com/UKPLab/elmo-bilstm-cnn-crf/blob/master/neuralnets/keraslayers/ChainCRF.py

- Make bi-directional a parameter?

- bert Different num layers (concat last four hidden?)
- bert + lstm?
- bert large?

- How to account for incorrect embedding type (i.e. nnlm) when running in graph mode? (also if I port elmo/bert etc to graph)
## Models to Add
- ELMo + LSTM/Bi-LSTM
- ELMo + LSTM/Bi-LSTM + Attn
- ELMo + Attn
- Albert 
- Distillbert
- https://github.com/AlexYangLi/TextClassification

## Misc
- Document the models in readme
- elmo needs to be strings (or tokenised strings - but those suck)
#### graph models
- elmo
- bert
- use
- mlstm_char_lm

## To try
- Visualising attention layers
- Saving trained embeddings