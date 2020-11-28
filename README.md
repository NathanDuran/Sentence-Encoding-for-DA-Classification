# Supervised and Semi-supervised Sentence Encoding Methods for Dialogue Act Classification

# TODO
- Document but remove use_crf from the run script OR update CRF to work on tf 1.15/2.X
add run_transformers notebook


## Misc
- Document the models in readme
- Document layers in readme
- Document datasets in readme
- Embeddings

#### graph models
Language models need to be strings to_tokens=false and input_shape=(1,)
- elmo (or tokenised strings - but those suck)
- albert
- bert (input_shape can be max_seq_length)
- use
- nnlm
- mlstm_char_lm
- ConveRT (requires linux (tensorflow_text))

## TF2/huggingface models
 Roberta
- GPT2
- DialoGPT
- XLNET