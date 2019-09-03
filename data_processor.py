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

    def __init__(self, set_name, output_dir, max_seq_length, vocab_size=None, to_tokens=True, pad_seq=True, to_lower=True, no_punct=False, label_index=2):
        """Constructs a DataProcessor for the specified dataset.

        Note: For MRDA data there is the option to choose which type of labelling is used.
        There are 3 different types: basic_labels, general_labels or full_labels.
        The label_index parameter is used to determine which index the labels can be found
        once the original data is split on the '|' character. If it is not specified then an index of 2 is used.

        Args:
            set_name (str): The name of this dataset can be any string but MUST include a substring from valid_set_names
            output_dir (str): Directory to save the processed data
            max_seq_length (int): Length to pad or truncate sentences to
            vocab_size (int): Specifies the size of the datasets vocabulary to use, if 'None' uses all words
            to_tokens (bool): Flag for tokenising input sentences, if false returns full sentence strings
            pad_seq (bool): Flag for padding sequences to max_seq_length
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
        self.vocab_size = vocab_size
        self.to_tokens = to_tokens
        self.pad_seq = pad_seq
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

    def get_val_examples(self):
        """Gets the Validation set from Github repository. Used to evaluate training.

        Returns:
             examples (list): A list of InputExamples for the training set
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Get the file from Github repo
            url = self.base_url + 'val_set.txt'
            urllib.request.urlretrieve(url, filename=temp_file)

            # Read lines and get examples
            with open(temp_file) as file:
                lines = [line.rstrip('\r\n') for line in file.readlines()]
        return self._get_examples(lines, "val")

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
        """Gets examples for the training, val and test sets from plain text files.

        Args:
            lines (list): List of str in the format <speaker>|<sentence>|<da-label>
            set_type (str): Specifies if this is the train, test or val dataset

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
                    else:
                        sentence_tokens = [token.orth_ for token in sentence_tokens]

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

    def get_dataset(self, to_numpy=False):
        """Helper function. Gets the metadata and all datasets from the Github repository and saves to file.

        Args:
            to_numpy (bool): Wheather to save dataset as numpy arrays or TFRecord files
        """

        vocabulary, labels = self.get_metadata()

        train_examples = self.get_train_examples()
        test_examples = self.get_test_examples()
        val_examples = self.get_val_examples()
        dev_examples = self.get_dev_examples()

        if to_numpy:
            self.convert_examples_to_numpy('train', train_examples, vocabulary, labels)
            self.convert_examples_to_numpy('test', test_examples, vocabulary, labels)
            self.convert_examples_to_numpy('val', val_examples, vocabulary, labels)
            self.convert_examples_to_numpy('dev', dev_examples, vocabulary, labels)
        else:
            self.convert_examples_to_record('train', train_examples, vocabulary, labels)
            self.convert_examples_to_record('test', test_examples, vocabulary, labels)
            self.convert_examples_to_record('val', val_examples, vocabulary, labels)
            self.convert_examples_to_record('dev', dev_examples, vocabulary, labels)

    def convert_examples_to_record(self, set_type, examples, vocabulary, labels):
        """Converts InputExamples to features and saves as TFRecord file.

        if to_tokens is True
            Tokenizes all text and strips whitespace.
            Converts to lowercase if to_lower=True.
            Removes punctuation if no_punct=True.
            Pads sentence with <unk> tokens to max_seq_length if pad_seq=True
            Converts sentence tokens to indices.

        Converts labels to indices.

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
            features['example_id'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[example_to_serialize.example_id.encode('utf-8')]))
            features['text'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=example_to_serialize.text)) if self.to_tokens else tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[example_to_serialize.text.encode('utf-8')]))
            features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=example_to_serialize.label))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            return tf_example.SerializeToString()

        print("Creating " + set_type + ".tf_record...")
        # Create TFRecord writer
        writer = tf.python_io.TFRecordWriter(os.path.join(self.output_dir, set_type + ".tf_record"))

        # Process each example and save to file
        for example in examples:

            if self.to_tokens:
                # Tokenize, convert to lowercase and remove punctuation
                tokens = tokenizer(example.text)
                if self.no_punct:
                    tokens = [token for token in tokens if not token.is_punct]
                if self.to_lower:
                    tokens = [token.orth_.lower() for token in tokens]
                else:
                    tokens = [token.orth_ for token in tokens]

                # Pad/truncate sequences to max_sequence_length (0 = <unk> token in vocabulary)
                if self.pad_seq:
                    tokens = [tokens[i] if i < len(tokens) else '<unk>' for i in range(self.max_seq_length)]

                # Convert word and label tokens to indices
                example.text = [vocabulary.token_to_idx[token] for token in tokens]

            # Convert labels to indices
            example.label = [labels.index(example.label)]

            # Serialize and write to TFRecord
            serialized_example = _serialize_example(example)

            writer.write(serialized_example)

    def convert_examples_to_numpy(self, set_type, examples, vocabulary, labels):
        """Converts InputExamples to features and saves as .npz file.

        if to_tokens is True
            Tokenizes all text and strips whitespace.
            Converts to lowercase if to_lower=True.
            Removes punctuation if no_punct=True.
            Pads sentence with <unk> tokens to max_seq_length if pad_seq=True
            Converts sentence tokens to indices.

        Converts labels to indices.

        Saves as .npz file.

         Args:
            set_type (str): Specifies if this is the training, validation or test data
            examples (list): List of InputExamples
            vocabulary (Gluonnlp Vocab): Datasets vocabulary
            labels (list): Datasets labels list
        """

        print("Creating " + set_type + ".npz...")
        examples_text = []
        examples_labels = []
        # Process each example and save to file
        for example in examples:

            if self.to_tokens:
                # Tokenize, convert to lowercase and remove punctuation
                tokens = tokenizer(example.text)
                if self.no_punct:
                    tokens = [token for token in tokens if not token.is_punct]
                if self.to_lower:
                    tokens = [token.orth_.lower() for token in tokens]
                else:
                    tokens = [token.orth_ for token in tokens]

                # Pad/truncate sequences to max_sequence_length (0 = <unk> token in vocabulary)
                if self.pad_seq:
                    tokens = [tokens[i] if i < len(tokens) else '<unk>' for i in range(self.max_seq_length)]

                # Convert word and label tokens to indices
                example.text = [vocabulary.token_to_idx[token] for token in tokens]

            # Convert labels to indices
            example.label = [labels.index(example.label)]

            # convert to numpy
            examples_text.append(example.text)
            examples_labels.append(example.label)

        # Save to npz
        examples_text = np.asarray(examples_text)
        examples_labels = np.asarray(examples_labels)
        np.savez_compressed(os.path.join(self.output_dir, set_type), text=examples_text, labels=examples_labels)

    def build_dataset_from_record(self, set_type, batch_size, repeat=None, is_training=True, drop_remainder=False):
        """Creates an iterable dataset from the specified TFRecord File
        
        Args:
            set_type (str): Specifies if this is the training, validation or test data
            batch_size (int): The number of examples per batch
            repeat (int): How many times the dataset with repeat until it is exhausted, if 'None' repeats forever
            is_training (bool): Flag determines if training set is shuffled
            drop_remainder (bool): Flag determines if last batch is dropped if not of batch_size
            
        Returns:
            dataset (TF Dataset): Iterable dataset of two tensors 'text' and 'label'
        """

        def _decode_single_record(serialized_example):
            """Decodes single TFRecord example into Tensors."""

            feature_map = {'example_id': tf.FixedLenFeature([], tf.string),
                           'text': tf.FixedLenFeature([self.max_seq_length], tf.int64)
                           if self.to_tokens else tf.FixedLenFeature([], tf.string),
                           'label': tf.FixedLenFeature([1], tf.int64)}

            # Parse the serialized example into a dictionary
            example = tf.parse_single_example(serialized_example, feature_map)

            # Get the tensor values from the dictionary
            text = tf.cast(example['text'], tf.int32) if self.to_tokens else example['text']
            label = tf.cast(example['label'], tf.int32)
            return text, label

        # Get the dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(os.path.join(self.output_dir, set_type + ".tf_record"))

        # For training, shuffle the data
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(repeat)
        dataset = dataset.map(lambda record: _decode_single_record(record))
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        return dataset

    def build_dataset_from_numpy(self, set_type, batch_size, is_training=True):
        """Creates an numpy dataset from the specified .npz File

        Args:
            set_type (str): Specifies if this is the training, validation or test data
            batch_size (int): The number of examples per batch
            is_training (bool): Flag determines if training set is shuffled

        Returns:
            text (np.array): Numpy array of input text
            labels (np.array: Numpy array of target labels
        """

        # Get the dataset from the .npz file
        dataset = np.load(os.path.join(self.output_dir, set_type + ".npz"))
        text = dataset['text']
        labels = dataset['labels']

        # For training, shuffle the data
        if is_training:
            combined = list(zip(text, labels))
            np.random.shuffle(combined)
            text, labels = zip(*combined)
            text = np.asarray(text)
            labels = np.asarray(labels)

        # Batch data
        text = list(batch(text, batch_size))
        labels = list(batch(labels, batch_size))
        text = np.asarray(text)
        labels = np.asarray(labels)

        return text, labels

    def build_dataset_for_bert(self, set_type, bert_tokenizer, batch_size, is_training=True):
        """Creates an numpy dataset for BERT from the specified .npz File

        Args:
            set_type (str): Specifies if this is the training, validation or test data
            bert_tokenizer (FullTokeniser): The BERT tokeniser
            batch_size (int): The number of examples per batch
            is_training (bool): Flag determines if training set is shuffled

        Returns:
            input_ids (np.array): Numpy array of BERT input ids
            input_masks (np.array): Numpy array of BERT input masks
            segment_ids (np.array): Numpy array of BERT segment ids
            labels (np.array): Numpy array of target labels
        """

        def _convert_single_example(bert_tokenizer, example_text, max_seq_length):
            """Converts a single sentence into BERT features."""

            text_tokens = bert_tokenizer.tokenize(example_text)
            if len(text_tokens) > max_seq_length - 2:
                text_tokens = text_tokens[0: (max_seq_length - 2)]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in text_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            return input_ids, input_mask, segment_ids
        """Converts a single sentence of text to BERT input features."""
        # Get the dataset from the .npz file
        dataset = np.load(os.path.join(self.output_dir, set_type + ".npz"))
        text = dataset['text']
        labels = dataset['labels']

        # Create BERT input features
        input_ids, input_masks, segment_ids = [], [], []
        for i in range(len(text)):
            input_id, input_mask, segment_id = _convert_single_example(bert_tokenizer, text[i], self.max_seq_length)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        # For training, shuffle the data
        if is_training:
            combined = list(zip(input_ids, input_masks, segment_ids, labels))
            np.random.shuffle(combined)
            input_ids, input_masks, segment_ids, labels = zip(*combined)
            input_ids = np.asarray(input_ids)
            input_masks = np.asarray(input_masks)
            segment_ids = np.asarray(segment_ids)
            labels = np.asarray(labels)

        # Batch data
        input_ids = list(batch(input_ids, batch_size))
        input_masks = list(batch(input_masks, batch_size))
        segment_ids = list(batch(segment_ids, batch_size))
        labels = list(batch(labels, batch_size))

        return np.asarray(input_ids), np.asarray(input_masks), np.asarray(segment_ids), np.asarray(labels)


def batch(input_arr, batch_size):
    """Yield successive batch_size chunks from input_arr."""
    for i in range(0, len(input_arr), batch_size):
        yield input_arr[i:i + batch_size]


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
