# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- test crf models?

- bert Different num layers (concat last four hidden?)
- bert + lstm?
- bert large?
- bert output? pooled, seq/mean_seq

- elmo try word_emb/elmo and default
- BiLSTM+ELMo+Attn?

- see if custom layers need loading in run_graph and run_bert? (and adding to init.py) (and creating get_config func, as in kmaxpooling layer)
## models

- albert?/ distillbert?

- CRF, More models and/or add use_crf to model_params (i.e. in model just swap classification layers)
- Make bi-directional a parameter?

- https://github.com/AlexYangLi/TextClassification



## Misc
- Document the models in readme

## To try
- Visualising attention layers
- Saving trained embeddings