import gluonnlp as nlp
import numpy as np


def get_embedding_processor(processor_type):
    """Utility function for returning an EmbeddingProcessor.

    Args:
        processor_type (str): The name of the EmbeddingProcessor

    Returns:
        processor (EmbeddingProcessor): An instance of the selected EmbeddingProcessor
    """
    embeddings = {'glove': GloveEmbedding(),
                  'word2vec': Word2VecEmbedding(),
                  'fasttext': FastTextEmbedding()}

    if processor_type.lower() not in embeddings.keys():
        raise Exception("The given embedding processor type: '" + processor_type + "' is not valid!\n" +
                        "Please select one from: " + str(list(embeddings.keys())) + " or create one.")
    else:
        return embeddings[processor_type]


class EmbeddingProcessor:
    """Contains function for mapping word tokens to embedding vectors"""

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


class GloveEmbedding(EmbeddingProcessor):

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

        # Get the specifies embedding file
        glove = nlp.embedding.GloVe(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(glove)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' does not match specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix


class Word2VecEmbedding(EmbeddingProcessor):

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

        # Get the specifies embedding file
        word2vec = nlp.embedding.Word2Vec(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(word2vec)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' does not match specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix


class FastTextEmbedding(EmbeddingProcessor):

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

        # Get the specifies embedding file
        fastext = nlp.embedding.FastText(source=embedding_source, embedding_root=embedding_dir)
        # Attach embeddings to the vocabulary
        vocabulary.set_embedding(fastext)

        # Check desired embedding dimensions is valid
        if len(vocabulary.embedding[0]) != embedding_dim:
            raise ValueError("The given embedding source dimension: '" + str(len(vocabulary.embedding[0])) +
                             "' does not match specified embedding dimensions: '" + str(embedding_dim) + "'.")

        # Generate empty numpy matrix of shape (vocabulary_size, embedding_dim)
        matrix = np.empty((len(vocabulary), embedding_dim), dtype='float32')

        # Copy vocabulary embeddings into matrix
        for i in range(len(vocabulary)):
            embedding = vocabulary.embedding[vocabulary.idx_to_token[i]].asnumpy()
            matrix[i] = embedding

        return matrix
