import os
import bz2
import gzip
import shutil
import tempfile
import gluonnlp as nlp
import numpy as np


def get_embedding_processor(processor_type):
    """Utility function for returning an EmbeddingProcessor.

    Args:
        processor_type (str): The name of the EmbeddingProcessor

    Returns:
        processor (EmbeddingProcessor): An instance of the selected EmbeddingProcessor
    """
    embeddings = {'random': Random(),
                  'glove': Glove(),
                  'word2vec': Word2Vec(),
                  'fasttext': FastText(),
                  'numberbatch': Numberbatch(),
                  'deps': Dependency(),
                  'nnlm': Default()}

    if processor_type.lower() not in embeddings.keys():
        raise Exception("The given embedding processor type: '" + processor_type + "' is not valid!\n" +
                        "Please select one from: " + str(list(embeddings.keys())) + " or create one.")
    else:
        return embeddings[processor_type]


class EmbeddingProcessor:
    """Contains function for mapping word tokens to embedding vectors."""

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        """Loads embeddings and maps word tokens to embedding vectors.

        Args:
            embedding_dir (str): The location to store and load embedding files
            embedding_source (string): Specifies which embedding source file to load
            embedding_dim (int): Length of vector to map tokens to, raises error if longer than loaded source files
            vocabulary (Gluonnlp Vocabulary): Maps word tokens to attached embedding vectors

        Returns:
            matrix (numpy array): A matrix of shape (vocabulary_size, embedding_dim) which maps words to embeddings
        """
        raise NotImplementedError()


class Default(EmbeddingProcessor):
    """Returns the default empty matrix for models that do not use embeddings."""

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        return []


class Random(EmbeddingProcessor):
    """Generates random embeddings matrix."""

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        np.random.seed(42)
        # Generate random numpy matrix of shape (vocabulary_size, embedding_dim) in range [-10, 10]
        matrix = 20 * np.random.random_sample((len(vocabulary), embedding_dim)) - 10
        # Set first two rows to 0, for <unk> and <pad> tokens
        matrix[0] = np.zeros(embedding_dim)
        matrix[1] = np.zeros(embedding_dim)
        return matrix


class Glove(EmbeddingProcessor):

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        """Loads GloVe embeddings and maps word tokens to embedding vectors.

        Valid GloVe source files:
        'glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d',
        'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d'
        """

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

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            embedding = [embedding[i] for i in range(embedding_dim)]
            matrix[i] = embedding

        return matrix


class Word2Vec(EmbeddingProcessor):

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        """Loads Word2Vec embeddings and maps word tokens to embedding vectors.

        Valid Word2Vec source files:
        'GoogleNews-vectors-negative300', 'freebase-vectors-skipgram1000-en', 'freebase-vectors-skipgram1000'
        """

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

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            embedding = [embedding[i] for i in range(embedding_dim)]
            matrix[i] = embedding

        return matrix


class FastText(EmbeddingProcessor):

    def get_embedding_matrix(self, embedding_dir, embedding_source, embedding_dim, vocabulary):
        """Loads FastText embeddings and maps word tokens to embedding vectors.

        Valid FastText source files:
        'crawl-300d-2M', 'crawl-300d-2M-subword', 'wiki.simple'
        """

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

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            embedding = [embedding[i] for i in range(embedding_dim)]
            matrix[i] = embedding

        return matrix


class Numberbatch(EmbeddingProcessor):
    """Generates ConceptNet Numberbatch embeddings matrix.
    Speer, R., Chin, J., & Havasi, C. (2016). ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.
    Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17) ConceptNet, 4444–4451.

    Embedding source available from: https://github.com/commonsense/conceptnet-numberbatch

    Valid FastText source files:
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

            # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
            matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

            # Copy vocabulary embeddings into matrix
            for i in range(len(vocabulary)):
                embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
                embedding = [embedding[i] for i in range(embedding_dim)]
                matrix[i] = embedding

        return matrix


class Dependency(EmbeddingProcessor):
    """Generates Dependency-based embeddings matrix.
    Levy, O., & Goldberg, Y. (2014). Dependency-Based Word Embeddings.
    Proceedings Ofthe 52nd Annual Meeting Ofthe Association for Computational Linguistics, 302–308.

    Embedding source available from: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

    Valid FastText source files:
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

            # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
            matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

            # Copy vocabulary embeddings into matrix
            for i in range(len(vocabulary)):
                embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
                embedding = [embedding[i] for i in range(embedding_dim)]
                matrix[i] = embedding

        return matrix
