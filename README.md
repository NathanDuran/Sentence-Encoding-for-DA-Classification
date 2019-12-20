# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?
- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)
https://github.com/UKPLab/elmo-bilstm-cnn-crf/blob/master/neuralnets/keraslayers/ChainCRF.py

- bert Different num layers (concat last four hidden?)
- bert + lstm?
- bert large?
- bert output? pooled, seq/mean_seq

- elmo try word_emb/elmo and default
- BiLSTM+ELMo+Attn?

- test load/save for eager models and bert

- Add new confusion matrix to comet experiment?

- Make bi-directional a parameter?
## models

- albert?/ distillbert?

- https://github.com/AlexYangLi/TextClassification

## Misc
- Document the models in readme
- elmo needs to be strings or tokenised strings

## To try
- Visualising attention layers
- Saving trained embeddings