import os
import pickle
import gluonnlp as nlp
from spacy.lang.en import English

# Initialise Spacy tokeniser
tokenizer = English().Defaults.create_tokenizer(English())


class InputExample(object):
    """A single training/test example for dialogue act classification classification."""

    def __init__(self, example_id, text, label):
        """Constructs an InputExample.

        Args:
          example_id (str): Unique id for the example.
          text (list): The tokenized text of the sentence
          label (str): The label of the example.
        """
        self.example_id = example_id
        self.text = text
        self.label = label

    def __repr__(self):
        return "ID: " + self.example_id + " Text: " + str(self.text) + " Label: " + self.label


class DataProcessor:
    """Converts sentences for dialogue act classification data sets.

    Loads plain text files as input in the format <speaker>|<sentence>|<da-label>.
    Tokenizes sentences, converts to lowercase, removes punctuation and creates list of input examples.

    Note: For MRDA data there is the option to choose which type of labelling is used.
    There are 3 different types basic_labels, general_labels or full_labels.
    The label_type parameter is used to determine which of these to use and at which index the labels can be found
    once the original data is split on the '|' character.
    If it is not specified then 'labels' is used to get labels from metadata and an index of 2 is used.

     Args:
            input_dir (str): Directory of the input data to be processed
            output_dir (str): Directory to save the processed
            max_sequence_length (int): Length to pad or truncate sentences to
            label_type (str): Determines which labels to use if there are more than one set (mostly for MRDA datset)

            embedding_type (str): Determines the type of embedding to be applied, if None embeddings are random
            embedding_dim (str): Determines size of embedding dimension, if None defaults to 100
    """

    def __init__(self, input_dir, output_dir, max_sequence_length, label_type='labels', embedding_type=None, embedding_dim=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_sequence_length = max_sequence_length
        self.label_type = label_type
        # self.embedding_type = embedding_type
        # self.embedding_dim = embedding_dim

        self.label_indexes = {'labels': 2,
                              'basic_labels': 2,
                              'general_labels': 3,
                              'full_labels': 4}

    def get_train_examples(self):
        """Training Set."""
        with open(self.input_dir + 'train_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "train")

    def get_eval_examples(self):
        """Evaluation Set. Set here WILL have labels and be used to evaluate training"""
        with open(self.input_dir + 'eval_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "eval")

    def get_test_examples(self):
        """Test Set. Set here will NOT have labels and be used to make predictions"""
        with open(self.input_dir + 'test_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._create_examples(lines, "test")

    def get_labels(self):
        """Load Labels from metadata. Is a GluonNLP Vocabulary object."""
        with open(self.input_dir + '/metadata/metadata.pkl', 'rb') as file:
            metadata = pickle.load(file)
        return metadata[self.label_type]

    def get_vocabulary(self):
        """Load the Vocabulary from the metadata. Is a GluonNLP Vocabulary object."""
        with open(self.input_dir + '/metadata/metadata.pkl', 'rb') as file:
            metadata = pickle.load(file)
        return metadata['vocabulary']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, eval and test sets from plain text files."""

        examples = []
        for (i, line) in enumerate(lines):
            example_id = set_type + "-" + str(i)

            # Split lines on '|' character to get raw sentences and labels
            sentence = line.split("|")[1]
            label = line.split("|")[self.label_indexes[self.label_type]]

            # Tokenize, convert to lowercase and remove punctuation
            sent_tokens = tokenizer(sentence)
            sent_tokens = [token.orth_.lower() for token in sent_tokens if not token.is_punct]

            # Create input example
            # TODO Don't add labels to test set?
            examples.append(InputExample(example_id=example_id, text=sent_tokens, label=label))
        return examples
