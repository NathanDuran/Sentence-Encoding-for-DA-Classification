import os
import urllib.request
import tempfile
import collections
import itertools
import pickle
from collections import Counter
import numpy as np
import tensorflow as tf
import gluonnlp as nlp
from spacy.lang.en import English

# Suppress TensorFlow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Initialise Spacy tokeniser
tokenizer = English().Defaults.create_tokenizer(English())


class InputExample(object):
    """A single training/test example for dialogue act classification."""

    def __init__(self, example_id, text, label):
        """Constructs an InputExample."""

        self.example_id = example_id
        self.text = text
        self.label = label

    def __repr__(self):
        return "ID: " + self.example_id + " Text: " + str(self.text) + " Label: " + str(self.label)


class DataProcessor:
    """Converts sentences for dialogue act classification into data sets."""

    def __init__(self, set_name,  output_dir, max_seq_length, pad_seq=True, vocab_size=None, to_lower=True, no_punct=True, label_index=2):
        """Constructs a DataProcessor for the specified dataset.

        Note: For MRDA data there is the option to choose which type of labelling is used.
        There are 3 different types: basic_labels, general_labels or full_labels.
        The label_index parameter is used to determine which index the labels can be found
        once the original data is split on the '|' character. If it is not specified then an index of 2 is used.

        Args:
            set_name (str): The name of this dataset can be any string but MUST include a substring from valid_set_names
            output_dir (str): Directory to save the processed data
            max_seq_length (int): Length to pad or truncate sentences to
            pad_seq (bool): Flag for padding sequences to max_seq_length
            vocab_size (int): Specifies the size of the datasets vocabulary to use, if 'None' uses all words
            to_lower (bool): Flag to convert words to lowercase
            no_punct (bool): Flag to remove punctuation from sentences
            label_index (int): Determines the label type is used if there is more than one type

        Attributes:
            metadata_file (str): Default metadata file location
            valid_set_names (list): List of the datsets this processor can create
            base_url (str): Url to the datsets Github repository folder
        """

        self.set_name = set_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.pad_seq = pad_seq
        self.vocab_size = vocab_size
        self.to_lower = to_lower
        self.no_punct = no_punct
        self.label_index = label_index

        self.metadata_file = os.path.join(self.output_dir, 'metadata.pkl')

        # Check the set name is valid
        self.valid_set_names = ['swda', 'mrda']
        if not any(substring in self.set_name.lower() for substring in self.valid_set_names):
            raise Exception("Specified dataset name: " + self.set_name + " is not valid! "
                            "Must contain a name from the following list: " + str(self.valid_set_names))

        # Set the download url
        self.base_url = ''
        if 'swda' in self.set_name.lower():
            self.base_url = 'https://raw.github.com/NathanDuran/Switchboard-Corpus/master/swda_data/'
        elif 'mrda' in self.set_name.lower():
            self.base_url = 'https://raw.github.com/NathanDuran/MRDA-Corpus/master/mrda_data/'

    def get_train_examples(self):
        """Gets the Training set from Github repository.

        Returns:
             examples (list): A list of InputExamples for the training set
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'train_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            # Read lines and get examples
            with open(temp_file) as file:
                lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "train")

    def get_eval_examples(self):
        """Gets the Evaluation set from Github repository. Used to evaluate training.

        Returns:
             examples (list): A list of InputExamples for the training set
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'eval_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            # Read lines and get examples
            with open(temp_file) as file:
                lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "eval")

    def get_test_examples(self):
        """Gets the Test set from Github repository. Used to make predictions.

        Returns:
             examples (list): A list of InputExamples for the training set
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'test_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            # Read lines and get examples
            with open(temp_file) as file:
                lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "test")

    def get_dev_examples(self):
        """Gets the Development set from Github repository. Smaller version of the training set.

        Returns:
             examples (list): A list of InputExamples for the training set
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'dev_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            # Read lines and get examples
            with open(temp_file) as file:
                lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "dev")

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
            examples.append(InputExample(example_id=example_id, text=sentence, label=label))
        return examples

    def get_metadata(self):
        """Generate a Vocabulary and label list from the whole dataset.

        Tokenizes all text and strips whitespace.
        Converts to lowercase if to_lower=True.
        Removes punctuation if no_punct=True.
        Keeps only vocab_size number of words.

        Counts labels and creates list of strings sorted in descending order of frequency

        Saves the vocabulary and labels to a pickle file.

        Returns:
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
            labels (list): Datasets labels
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'full_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            with open(temp_file) as file:
                label_counter = []
                tokenized_utterances = []
                for line in file:
                    # Get the labels
                    label_counter.append(line.split('|')[self.label_index].rstrip('\r\n'))
                    # Get sentence text
                    sentence = line.split('|')[1].rstrip('\r\n')

                    # Tokenize, convert to lowercase and remove punctuation
                    sentence_tokens = tokenizer(sentence)
                    if self.no_punct:
                        sentence_tokens = [token for token in sentence_tokens if not token.is_punct]
                    if self.to_lower:
                        sentence_tokens = [token.orth_.lower() for token in sentence_tokens]

                    tokenized_utterances.append(sentence_tokens)

            # Count the word frequencies and generate vocabulary with vocabulary_size (-1 to account for <unk>)
            vocab_counter = nlp.data.count_tokens(list(itertools.chain(*tokenized_utterances)))
            vocabulary = nlp.Vocab(vocab_counter, self.vocab_size - 1,
                                   padding_token=None,
                                   bos_token=None,
                                   eos_token=None)

            # Create and sort the labels counter
            label_counter = Counter(label_counter)
            labels = sorted(label_counter, key=label_counter.get, reverse=True)

            # Create the metadata dictionary
            metadata = {'labels': labels,
                        'vocabulary': vocabulary}
            # Save to pickle
            with open(self.metadata_file, 'wb') as file:
                pickle.dump(metadata, file, protocol=2)

        return vocabulary, labels

    def load_metadata(self):
        """Load Vocabulary and Labels from metadata file.

        Returns:
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
            labels (list): Datasets labels list
        """

        if not os.path.isfile(self.metadata_file):
            raise FileNotFoundError("Metadata has not been created yet! "
                                    "Must call get_metadata() to generate and save to a file first!")

        with open(self.metadata_file, 'rb') as file:
            metadata = pickle.load(file)
        return metadata['vocabulary'], metadata['labels']

    def load_labels(self):
        """Load Labels from metadata file.

        Returns:
            labels (list): Datasets labels list
        """

        if not os.path.isfile(self.metadata_file):
            raise FileNotFoundError("Metadata has not been created yet! "
                                    "Must call get_metadata() to generate and save to a file first!")

        with open(self.metadata_file, 'rb') as file:
            metadata = pickle.load(file)
        return metadata['labels']

    def load_vocabulary(self):
        """Load Vocabulary from metadata file.

        Returns:
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
        """

        if not os.path.isfile(self.metadata_file):
            raise FileNotFoundError("Metadata has not been created yet! "
                                    "Must call get_metadata() to generate and save to a file first!")

        with open(self.metadata_file, 'rb') as file:
            metadata = pickle.load(file)
        return metadata['vocabulary']

    def get_dataset(self):
        """Helper function. Gets the metadata and all datasets from the Github repository and saves to file."""

        vocabulary, labels = self.get_metadata()

        train_examples = self.get_train_examples()
        self.convert_examples_to_record('train', train_examples, vocabulary, labels)

        test_examples = self.get_test_examples()
        self.convert_examples_to_record('test', test_examples, vocabulary, labels)

        eval_examples = self.get_eval_examples()
        self.convert_examples_to_record('eval', eval_examples, vocabulary, labels)

        dev_examples = self.get_dev_examples()
        self.convert_examples_to_record('dev', dev_examples, vocabulary, labels)

    def convert_examples_to_record(self, set_type, examples, vocabulary, labels):
        """Converts InputExamples to features and saves as TFRecord file.

        Tokenizes all text and strips whitespace.
        Converts to lowercase if to_lower=True.
        Removes punctuation if no_punct=True.
        Pads sentence with <unk> tokens to max_seq_length if pad_seq=True

        Converts sentence tokens and labels to indices.

        Saves as TFRecord file.

         Args:
            set_type (str): Specifies if this is the training, validation or test data
            examples (list): List of InputExamples
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
            labels (list): Datasets labels list
        """

        def _serialize_example(example_to_serialize):
            """Converts an InputExample into a serialized format for TFRecords"""
            features = collections.OrderedDict()
            # Strings must be encoded to bytes
            features['example_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[example_to_serialize.example_id.encode('utf-8')]))
            features['text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=example_to_serialize.text))
            features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=example_to_serialize.label))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            return tf_example.SerializeToString()

        print("Creating " + set_type + ".tf_record...")
        # Create TFRecord writer
        writer = tf.python_io.TFRecordWriter(os.path.join(self.output_dir, set_type + ".tf_record"))

        # Process each example and save to file
        for example in examples:

            # Tokenize, convert to lowercase and remove punctuation
            tokens = tokenizer(example.text)
            if self.no_punct:
                tokens = [token for token in tokens if not token.is_punct]
            if self.to_lower:
                tokens = [token.orth_.lower() for token in tokens]

            # Pad/truncate sequences to max_sequence_length (0 = <unk> token in vocabulary)
            if self.pad_seq:
                tokens = [tokens[i] if i < len(tokens) else '<unk>' for i in range(self.max_seq_length)]

            # Convert word and label tokens to indices
            example.text = [vocabulary.token_to_idx[token] for token in tokens]
            example.label = [labels.index(example.label)]

            # Serialize and write to TFRecord
            serialized_example = _serialize_example(example)

            writer.write(serialized_example)

    def build_dataset_from_record(self, set_type, batch_size, repeat=None, is_training=True, drop_remainder=False):
        """Creates an iterable dataset from the specified TFRecord File
        
        Args:
            set_type (str): Specifies if this is the training, validation or test data
            batch_size (int): The number of examples per batch
            repeat (int): How many times the dataset with repeat untill it is exhausted, if 'None' repeats forever
            is_training (bool): Flag determines if training set is shuffled
            drop_remainder (bool): Flag determines if last batch is dropped if not of batch_size
            
        Returns:
            dataset (TF Dataset): Iterable dataset of two tensors 'text' and 'label'
        """

        def _decode_single_record(serialized_example):
            """Decodes single TFRecord example into Tensors."""

            feature_map = {
                "example_id": tf.FixedLenFeature([], tf.string),
                "text": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "label": tf.FixedLenFeature([1], tf.int64),
            }

            # Parse the serialized example into a dictionary
            example = tf.parse_single_example(serialized_example, feature_map)

            # Get the tensor values from the dictionary
            text = tf.cast(example['text'], tf.int32)
            label = tf.cast(example['label'], tf.int32)
            return text, label

        # Get the dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(os.path.join(self.output_dir, set_type + ".tf_record"))

        # For training, we want a lot of parallel reading and shuffling.
        # For testing, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat(repeat)
        dataset = dataset.map(lambda record: _decode_single_record(record))
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        return dataset


def to_one_hot(label, labels):
    """Converts label string representation into one-hot encoded list."""

    # Create zeros array same length as labels
    one_hot = np.zeros(len(labels), dtype='int32')

    # Set the labels index to 1
    one_hot[labels.index(label)] = 1

    return one_hot


def from_one_hot(one_hot, labels):
    """Converts one-hot encoded label list into its string representation."""
    return labels[int(np.argmax(one_hot))]
