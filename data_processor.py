import os
import collections
import itertools
import numpy as np
import tensorflow as tf
import gluonnlp as nlp
from spacy.lang.en import English

# Initialise Spacy tokeniser
tokenizer = English().Defaults.create_tokenizer(English())


class InputExample(object):
    """A single unprocessed training/test example for dialogue act classification."""

    def __init__(self, example_id, text, label):
        """Constructs an InputExample."""

        self.example_id = example_id
        self.text = text
        self.label = label

    def __repr__(self):
        return "ID: " + self.example_id + " Text: " + str(self.text) + " Label: " + str(self.label)


class DataProcessor:
    """Converts sentences for dialogue act classification into data sets."""

    def __init__(self, set_name, input_dir, output_dir, max_seq_length, vocab_size=None, label_index=2):
        """Constructs a DataProcessor for the specified dataset

        Note: For MRDA data there is the option to choose which type of labelling is used.
        There are 3 different types basic_labels, general_labels or full_labels.
        The label_index parameter is used to determine which index the labels can be found
        once the original data is split on the '|' character. If it is not specified then an index of 2 is used.

        Args:
            set_name (str): The name of this dataset
            input_dir (str): Directory of the input data to be processed
            output_dir (str): Directory to save the processed data
            max_seq_length (int): Length to pad or truncate sentences to
            vocab_size (int): Specifies the size of the datasets vocabulary to use, if 'None' uses all words
            label_index (int): Determines the label type is used if there is more than one set

        Attributes:
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
            labels (list): Datasets labels list
            embeddings (dict): Maps embedding_type to embedding processor
        """

        self.set_name = set_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.vocabulary_size = vocab_size
        self.label_index = label_index

        # Get the datasets vocabulary and label lists
        self.vocabulary = self.get_vocabulary()
        self.labels = self.get_labels()

        self.embeddings = {'glove': GloveEmbedding(),
                           'word2vec': Word2VecEmbedding(),
                           'fasttext': FastTextEmbedding()}

    def get_train_examples(self):
        """Training Set."""
        with open(self.input_dir + 'train_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "train")

    def get_eval_examples(self):
        """Evaluation Set. Set here WILL have labels and be used to evaluate training"""
        with open(self.input_dir + 'eval_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "eval")

    def get_test_examples(self):
        """Test Set. Set here will NOT have labels and be used to make predictions"""
        with open(self.input_dir + 'test_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "test")

    def get_dev_examples(self):
        """Dev Set. Smaller version of the training set."""
        with open(self.input_dir + 'dev_set.txt', "r") as file:
            # Read a line and strip newline char
            lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "dev")

    def get_labels(self):
        """Load Labels from metadata text file."""
        with open(self.input_dir + '/metadata/labels.txt', 'r') as file:
            labels = [line.split()[0] for line in file.readlines()]
        return labels

    def get_vocabulary(self):
        """Generate a Vocabulary from the whole dataset.
        Tokenizes, strips whitespace and lower-cases text to create a GluonNLP Vocabulary object.
        """
        with open(self.input_dir + '/all_' + self.set_name + '.txt', 'r') as file:

            tokenized_utterances = []
            for line in file:
                # Get text and strip whitespace
                sentence = line.split('|')[1].rstrip('\r\n')
                # Tokenize, convert to lowercase and remove punctuation
                sentence_tokens = tokenizer(sentence)
                sentence_tokens = [token.orth_.lower() for token in sentence_tokens if not token.is_punct]

                tokenized_utterances.append(sentence_tokens)

            # Count the word frequencies and generate vocabulary with vocabulary_size (-1 to account for <unk>)
            vocab_counter = nlp.data.count_tokens(list(itertools.chain(*tokenized_utterances)))
            vocabulary = nlp.Vocab(vocab_counter, self.vocabulary_size - 1, padding_token=None, bos_token=None, eos_token=None)

        return vocabulary

    def _get_examples(self, lines, set_type):
        """Gets examples for the training, eval and test sets from plain text files.

        Args:
            lines (list): List of str in the format <speaker>|<sentence>|<da-label>
            set_type (str): Specifies if this is the train, test or eval dataset

        Returns:
            examples (list): A list of InputExamples
        """

        examples = []
        for (i, line) in enumerate(lines):
            # Set a unique example ID
            example_id = set_type + "-" + str(i)

            # Split lines on '|' character to get raw sentences and labels
            sentence = line.split("|")[1]
            label = line.split("|")[self.label_index]

            # Create input example
            # TODO Don't add labels to test set?
            examples.append(InputExample(example_id=example_id, text=sentence, label=label))
        return examples

    def _serialize_example(self, example):

        features = collections.OrderedDict()
        features['example_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[example.example_id.encode('utf-8')]))
        features['text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=example.text))
        features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[example.label]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        return tf_example.SerializeToString()

    def convert_examples_to_features(self, examples, set_type, max_seq_length):
        """Converts InputExamples to features and saves as TFRecord file.

         Args:
                examples (list): List of InputExamples
                set_type (str): Specifies if this is the training, validation or test data
                max_seq_length (int): Length to pad or truncate sentences to
        """

        print("Creating " + set_type + ".tf_record")
        # Create TFRecord writer
        writer = tf.python_io.TFRecordWriter(os.path.join(self.output_dir, set_type + ".tf_record"))

        # Process each example and save to file
        for example in examples:

            # Tokenize, convert to lowercase and remove punctuation
            sentence_tokens = tokenizer(example.text)
            sentence_tokens = [token.orth_.lower() for token in sentence_tokens if not token.is_punct]

            # Pad/truncate sequences to max_sequence_length (0 = <unk> token in vocabulary)
            padded_sentence = [sentence_tokens[i] if i < len(sentence_tokens) else '<unk>' for i in range(max_seq_length)]

            # Convert word and label tokens to indices
            example.text = [self.vocabulary.token_to_idx[token] for token in padded_sentence]
            example.label = self.labels.index(example.label)

            # Serialize and write to TFRecord
            serialized_example = self._serialize_example(example)
            writer.write(serialized_example)

    def convert_all_examples(self):
        """Converts all datasets into features and saves as TFRecord file."""

        train_examples = self.get_train_examples()
        self.convert_examples_to_features(train_examples, 'train', self.max_seq_length)

        eval_examples = self.get_eval_examples()
        self.convert_examples_to_features(eval_examples, 'eval', self.max_seq_length)

        test_examples = self.get_test_examples()
        self.convert_examples_to_features(test_examples, 'test', self.max_seq_length)

        dev_examples = self.get_dev_examples()
        self.convert_examples_to_features(dev_examples, 'dev', self.max_seq_length)

    def get_embedding_matrix(self, embedding_type, embedding_source, embedding_dim):
        """Gets the specifies embedding type and creates an embedding matrix.

        Args:
            embedding_type (str): Specifies the type of embedding to use
            embedding_source (string): Specifies which embedding source file to load
            embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files

        Returns:
            embedding_matrix (numpy array): A matrix of dimension (vocabulary size, embedding dimension)
        """

        # Get the embedding processor
        embedding_processor = self.embeddings[embedding_type]
        embedding_matrix = embedding_processor.get_embeddings(self.vocabulary, embedding_source, embedding_dim)
        return embedding_matrix

    def to_one_hot(self, label, labels):
        """Converts label into one-hot encoding

        Args:
            label (str): The labels token string
            labels (list): List of datasets labels

        Returns:
            one_hot (list): A one hot encoded encoding of the label
        """

        # Create zeros array same length as labels
        one_hot = np.zeros(len(labels), dtype='int32')

        # Set the labels index to 1
        one_hot[labels.index(label)] = 1

        return one_hot

    def from_one_hot(self, one_hot, labels):
        """Converts one-hot encoded label into string

        Args:
            one_hot (list): The label as one-hot vector
            labels (list): List of datasets labels

        Returns:
            label (str): The labels string representation
        """

        return labels[int(np.argmax(one_hot))]


class EmbeddingProcessor(object):
    def __init__(self):
        """Embedding Processor base class.

        Attributes:
            embedding_dir (str): Specifies where embedding files will be downloaded and/or stored
        """
        self.embedding_dir = 'embeddings/'

    def get_embeddings(self, vocabulary, embedding_source, embedding_dim):
        """Loads embeddings and maps word tokens to embedding vectors.

        Args:
            vocabulary (Gluonnlp Vocabulary): Maps word tokens to attached embedding vectors
            embedding_source (string): Specifies which embedding source file to load
            embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files
        """
        raise NotImplementedError()


class GloveEmbedding(EmbeddingProcessor):

    def get_embeddings(self, vocabulary, embedding_source, embedding_dim):
        """Loads GloVe embeddings and maps word tokens to embedding vectors.

        Valid GloVe source files:
        'glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d',
        'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d'
        """

        # Get the specifies embedding file
        glove = nlp.embedding.GloVe(source=embedding_source, embedding_root=self.embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(glove)

        # Determine desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("Embedding source dimensions do not match specified embedding dimensions!")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix


class Word2VecEmbedding(EmbeddingProcessor):

    def get_embeddings(self, vocabulary, embedding_source, embedding_dim):
        """Loads Word2Vec embeddings and maps word tokens to embedding vectors.

        Valid Word2Vec source files:
        'GoogleNews-vectors-negative300', 'freebase-vectors-skipgram1000-en', 'freebase-vectors-skipgram1000'
        """

        # Get the specifies embedding file
        glove = nlp.embedding.Word2Vec(source=embedding_source, embedding_root=self.embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(glove)

        # Determine desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("Embedding source dimensions do not match specified embedding dimensions!")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix


class FastTextEmbedding(EmbeddingProcessor):

    def get_embeddings(self, vocabulary, embedding_source, embedding_dim):
        """Loads FastText embeddings and maps word tokens to embedding vectors.

        Valid FastText source files:
        'crawl-300d-2M', 'crawl-300d-2M-subword', 'wiki.simple'
        """

        # Get the specifies embedding file
        glove = nlp.embedding.FastText(source=embedding_source, embedding_root=self.embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(glove)

        # Determine desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("Embedding source dimensions do not match specified embedding dimensions!")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix

