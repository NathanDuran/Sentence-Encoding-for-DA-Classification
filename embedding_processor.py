import os
import bz2
import gzip
import shutil
import tempfile
import gluonnlp as nlp
import numpy as np
from abc import ABC, abstractmethod


def get_embedding(embedding_dir, embedding_type, embedding_source=None, embedding_dim=300, vocabulary=None):
    """Utility function for returning an embedding matrix generated by an EmbeddingProcessor.
    Valid embedding types:
    'char', 'random', 'glove', 'word2vec', 'fasttext', 'numberbatch', 'deps'

    Args:
        embedding_dir (str): The location to store and load embedding files. If it doesn't exist it will be created.
        embedding_type (str): The name of the EmbeddingProcessor.
        embedding_source (string): Specifies which embedding source file to load, or None for char embeddings.
        embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files.
        vocabulary (Gluonnlp Vocabulary): Maps word tokens to attached embedding vectors, or None for char embeddings.

    Attributes:
        embedding_types (dict): Dictionary mapping embedding_type strings to EmbeddingProcessor class.

    Returns:
        embedding_matrix (numpy array): A matrix of shape (vocabulary_size, embedding_dim) mapping words to embeddings.
    """
    embedding_types = {'char': Character(),
                       'random': Random(),
                       'glove': Glove(),
                       'word2vec': Word2Vec(),
                       'fasttext': FastText(),
                       'numberbatch': Numberbatch(),
                       'deps': Dependency()}

    if not os.path.exists(embedding_dir):
        print("Creating the embedding directory " + embedding_dir)
        os.makedirs(embedding_dir)

    if embedding_type.lower() in embedding_types.keys():
        processor = embedding_types[embedding_type.lower()]
        return processor.get_embedding_matrix(embedding_dir, embedding_source, embedding_dim, vocabulary)
    else:
        raise Exception("The given embedding type: '" + embedding_type + "' is not valid!\n" +
                        "Please select one from: " + str(list(embedding_types.keys())) + " or create one.")


class EmbeddingProcessor(ABC):
    """EmbeddingProcessor abstract class. Contains function for mapping word tokens to embedding vectors."""

    @abstractmethod
    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        """Loads embeddings and maps word tokens to embedding vectors.

        Args:
            embedding_dir (str): The location to store and load embedding files.
            embedding_source (string): Specifies which embedding source file to load.
            embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files.
            vocabulary (Gluonnlp Vocabulary): Maps word tokens to attached embedding vectors.

        Returns:
            embedding_matrix (numpy array): A matrix of shape (vocabulary_size, embedding_dim) mapping words to embeddings.
        """
        raise NotImplementedError()

    @staticmethod
    def copy_embedding_to_matrix(embedding_dim, vocabulary):
        """Copies embeddings that have been attached to a Gluonnlp Vocabulary into a numpy matrix.
        The vocabulary <pad> and <unk> tokens are set to 0's.
        Words that appear in the vocabulary but not in the original embedding are randomly generated.

        Args:
            embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files.
            vocabulary (Gluonnlp Vocabulary): Maps word tokens to attached embedding vectors.

        Returns:
            embedding_matrix (numpy array): A matrix of shape (vocabulary_size, embedding_dim) mapping words to embeddings.
        """
        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        embedding_matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            # Get the current word
            current_word = vocabulary.idx_to_token[i]

            # If it is the <pad>/<unk> token set to 0's, else get the embedding
            if current_word == vocabulary.unknown_token or current_word == vocabulary.padding_token:
                word_embedding = np.zeros(embedding_dim)
            else:
                word_embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
                # If the words embedding is not in the original embeddings set to random
                if np.count_nonzero(word_embedding) == 0:
                    word_embedding = np.random.uniform(low=-1, high=1, size=embedding_dim)

            # Add embedding to matrix
            embedding_matrix[i] = word_embedding[:embedding_dim]
        return embedding_matrix


class Character(EmbeddingProcessor):
    """Generates random character embeddings matrix with values in in the range [-1, 1].

    Uses the ELMo special character vocabulary. Specifically, char ids 0-255 come from utf-8 encoding bytes.
    Above 256 are reserved for special tokens:

    <bos> (256) – The index of beginning of the sentence character is 256 in ELMo.

    <eos> (257) – The index of end of the sentence character is 257 in ELMo.

    <bow> (258) – The index of beginning of the word character is 258 in ELMo.

    <eow> (259) – The index of end of the word character is 259 in ELMo.

    <pad> (260) – The index of padding character is 260 in ELMo. Encoded as 0's.
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        vocabulary_size = 261
        special_tokens = {'<bos>': 256, '<eos>': 257, '<bow>': 258, '<eow>': 259, '<pad>': 260}

        # Generate random numpy matrix of shape (vocabulary_size, embedding_dim) in range [-1, 1]
        embedding_matrix = np.random.uniform(low=-1, high=1, size=(vocabulary_size, embedding_dim))

        # Set the <pad> token to 0's
        embedding_matrix[special_tokens['<pad>']] = np.zeros(embedding_dim)
        return embedding_matrix


class Random(EmbeddingProcessor):
    """Generates random embeddings matrix with values in in the range [-1, 1]."""

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        # Generate random numpy matrix of shape (vocabulary_size, embedding_dim) in range [-1, 1]
        embedding_matrix = np.random.uniform(low=-1, high=1, size=(len(vocabulary), embedding_dim))

        # Set the <pad>/<unk> tokens to 0's
        embedding_matrix[vocabulary.token_to_idx[vocabulary.padding_token]] = np.zeros(embedding_dim)
        embedding_matrix[vocabulary.token_to_idx[vocabulary.unknown_token]] = np.zeros(embedding_dim)
        return embedding_matrix


class Glove(EmbeddingProcessor):
    """Generates GloVe embedding matrix.
    Pennington, J., Socher, R. and Manning, C.D. (2014) GloVe: Global Vectors for Word Representation.
    In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).

    Valid GloVe source files:
    'glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d',
    'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d'
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        valid_source_files = ['glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d',
                              'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d',
                              'glove.twitter.27B.100d', 'glove.twitter.27B.200d',
                              'glove.twitter.27B.25d', 'glove.twitter.27B.50d']

        # Check source file is valid
        if embedding_source not in valid_source_files:
            raise Exception("The given embedding source file: '" + embedding_source + "' is not valid!\n" +
                            "Please select one from: " + str(valid_source_files))

        # Get the specified embedding file
        glove = nlp.embedding.GloVe(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(glove)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) < embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' less than specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate embedding matrix of shape (vocabulary_size, embedding_dim)
        embedding_matrix = self.copy_embedding_to_matrix(embedding_dim, vocabulary)

        return embedding_matrix


class Word2Vec(EmbeddingProcessor):
    """Generates Word2Vec embedding matrix.
    Mikolov, T., Yih, W.-T. and Zweig, G. (2013) Linguistic Regularities in Continuous Space Word Representations.
    Proceedings of NAACL-HLT [online]. (June), pp. 746–751.

    Valid Word2Vec source files:
    'GoogleNews-vectors-negative300', 'freebase-vectors-skipgram1000-en', 'freebase-vectors-skipgram1000'
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        valid_source_files = ['GoogleNews-vectors-negative300',
                              'freebase-vectors-skipgram1000-en',
                              'freebase-vectors-skipgram1000']

        # Check source file is valid
        if embedding_source not in valid_source_files:
            raise Exception("The given embedding source file: '" + embedding_source + "' is not valid!\n" +
                            "Please select one from: " + str(valid_source_files))

        # Get the specified embedding file
        word2vec = nlp.embedding.Word2Vec(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(word2vec)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) < embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' less than specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate embedding matrix of shape (vocabulary_size, embedding_dim)
        embedding_matrix = self.copy_embedding_to_matrix(embedding_dim, vocabulary)

        return embedding_matrix


class FastText(EmbeddingProcessor):
    """Generates FastText embedding matrix.
    Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T. (2017) Bag of Tricks for Efficient Text Classification.
    In: the Association for Computational Linguistics [online]. 2017 Valencia, Spain: ACL. pp. 427–431.

    Valid FastText source files:
    'crawl-300d-2M', 'crawl-300d-2M-subword', 'wiki.simple'
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        valid_source_files = ['crawl-300d-2M', 'crawl-300d-2M-subword', 'wiki.simple']

        # Check source file is valid
        if embedding_source not in valid_source_files:
            raise Exception("The given embedding source file: '" + embedding_source + "' is not valid!\n" +
                            "Please select one from: " + str(valid_source_files))

        # Get the specified embedding file
        fastext = nlp.embedding.FastText(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(fastext)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) < embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' less than specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate embedding matrix of shape (vocabulary_size, embedding_dim)
        embedding_matrix = self.copy_embedding_to_matrix(embedding_dim, vocabulary)

        return embedding_matrix


class Numberbatch(EmbeddingProcessor):
    """Generates ConceptNet Numberbatch embeddings matrix.
    Speer, R., Chin, J., & Havasi, C. (2016). ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.
    Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17) ConceptNet, 4444–4451.

    Embedding source available from: https://github.com/commonsense/conceptnet-numberbatch

    Valid Numberbatch source files:
    'numberbatch-en'
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        valid_source_files = ['numberbatch-en']

        source_file = os.path.join(embedding_dir, 'numberbatch', embedding_source + '-19.08.gz')

        # Check source file is valid
        if embedding_source not in valid_source_files or not os.path.exists(source_file):
            raise Exception("The given embedding source file: '" + embedding_source + "' is not valid!\n" +
                            "Please select one from: " + str(valid_source_files))

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=embedding_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Unzip and copy to temp directory
            with gzip.open(source_file, 'rb') as file_in:
                with open(temp_file, 'wb') as file_out:
                    shutil.copyfileobj(file_in, file_out)

            # Get the specified embedding file
            numberbatch = nlp.embedding.TokenEmbedding.from_file(file_path=temp_file, elem_delim=' ')
            # Attach embeddings to the vocabulary
            vocabulary.set_embedding(numberbatch)

            # Check desired embedding dimensions is valid
            if len(vocabulary.embedding[0]) < embedding_dim:
                raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                                 "' less than specified embedding dimensions: '" + str(embedding_dim) + "'.")

            # Generate embedding matrix of shape (vocabulary_size, embedding_dim)
            embedding_matrix = self.copy_embedding_to_matrix(embedding_dim, vocabulary)

        return embedding_matrix


class Dependency(EmbeddingProcessor):
    """Generates Dependency-based embeddings matrix.
    Levy, O., & Goldberg, Y. (2014). Dependency-Based Word Embeddings.
    Proceedings Of the 52nd Annual Meeting Of the Association for Computational Linguistics, 302–308.

    Embedding source available from: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

    Valid Dependency source files:
    'deps'
    """

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):

        valid_source_files = ['deps']

        source_file = os.path.join(embedding_dir, 'dependency', embedding_source + '.words.bz2')

        # Check source file is valid
        if embedding_source not in valid_source_files or not os.path.exists(source_file):
            raise Exception("The given embedding source file: '" + embedding_source + "' is not valid!\n" +
                            "Please select one from: " + str(valid_source_files))

        # Create a temporary directory
        with tempfile.TemporaryDirectory(dir=embedding_dir) as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'temp')

            # Unzip and copy to temp directory
            with bz2.BZ2File(source_file) as file_in, open(temp_file, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)

            # Get the specified embedding file
            dependency = nlp.embedding.TokenEmbedding.from_file(file_path=temp_file, elem_delim=' ')
            # Attach embeddings to the vocabulary
            vocabulary.set_embedding(dependency)

            # Check desired embedding dimensions is valid
            if len(vocabulary.embedding[0]) < embedding_dim:
                raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                                 "' less than specified embedding dimensions: '" + str(embedding_dim) + "'.")

            # Generate embedding matrix of shape (vocabulary_size, embedding_dim)
            embedding_matrix = self.copy_embedding_to_matrix(embedding_dim, vocabulary)

        return embedding_matrix
