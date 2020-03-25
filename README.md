# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?
- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)
https://github.com/UKPLab/elmo-bilstm-cnn-crf/blob/master/neuralnets/keraslayers/ChainCRF.py

- Make bi-directional a parameter?

- mlstm_char_lm seq length / output mode? Can I run it in a loop?

## Models to Add
- ELMo + LSTM/Bi-LSTM
- ELMo + LSTM/Bi-LSTM + Attn
- ELMo + Attn
- BERT + LSTM
- BERT Large
- Albert 
- Distillbert
- https://github.com/AlexYangLi/TextClassification

## Misc
- Document the models in readme

Language models need to be strings to_tokens=false
- elmo: (or tokenised strings - but those suck)
- use:
- mlstm_char_lm:

#### graph models
- elmo
- bert
- use
- mlstm_char_lm

## To try
- Visualising attention layers
- Saving trained embeddings