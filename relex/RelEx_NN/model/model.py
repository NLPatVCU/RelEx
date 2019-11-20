# Author : Samantha Mahendran for RelEx

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os, logging, tempfile
import psutil
from segment import SetConnection

def create_validation_data(train_data, train_label, num_data=1000):
    """
    Splits the input data into training and validation. By default it takes first 1000 as the validation.

    :param num_data: number of files split as validation data
    :param train_label: list of the labels of the training data
    :param train_data: list of the training data
    :return:train samples, validation samples
    """

    x_val = train_data[:num_data]
    x_train = train_data[num_data:]

    y_val = train_label[:num_data]
    y_train = train_label[num_data:]

    return x_train, x_val, y_train, y_val


class Model:

<<<<<<< HEAD
    def __init__(self, data_object, segment=True, test=False, multilabel=True, one_hot=False, common_words=10000, maxlen=100):
=======
    def __init__(self, sentences, labels, preceding_segs, concept1_segs, middle_segs, concept2_segs, succeeding_segs, rel_labels, no_labels, dataset, CSV=True segment=True, test=False, multilabel=True, one_hot=False, common_words=10000, maxlen=100 ):
>>>>>>> 8c3c10cca94e4f0182297c5a742e3ec3edb4d11e
        """
        :param data_object: call set_connection here
        :param segment: Flag to be set to activate segment-CNN (default-True)
        :param test: Flag to be set to validate the model on the test dataset (default-False)
        :param one_hot: Flag to be set to create one-hot vectors (default-False)
        :param common_words: Number of words to consider as features (default = 10000)
        :param maxlen: maximum length of the vector (default = 100)
        :param sentences: path to sentences
        :param labels: path to labels
        :param preceding_segs: path to preceding segements
        :param concept1_segs: path to concpet 1 segements
        :param middle_segs: path to middle segements
        :param concept2_segs: path to concept2 segements
        :param succeeding_segs: path to succeeding segements
        :param rel_labels: array of relationship labels
        :param no_labels: label to be used when there is no relationship

        """
        self.one_hot = one_hot
        self.segment = segment
        self.test = test
        self.multilabel = multilabel
        self.common_words = common_words
        self.maxlen = maxlen
<<<<<<< HEAD
        self.data_object = data_object

        # read dataset from external files
        train_data = data_object['sentence']
        train_labels = data_object['label']
        print(train_data)
        print(train_labels)

        if self.test:
            test_data = data_object['sentence']
            test_labels = data_object['label']
=======
        self.CSV = CSV
        if self.CSV:
            self.sentences = sentences
            self.labels = labels
            self.preceding_segs = preceding_segs
            self.concept1_segs = concept1_segs
            self.middle_segs = middle_segs
            self.concept2_segs = concept2_segs
            self.succeeding_segs = succeeding_segs
            self.dataset = None
        else:
            self.sentences = None
            self.labels = None
            self.preceding_segs = None
            self.concept1_segs = None
            self.middle_segs = None
            self.concept2_segs = None
            self.succeeding_segs = None
            self.dataset = dataset

        self.rel_labels = rel_labels
        self.no_labels = no_labels

        train_data = SetConnection(self.dataset, self.rel_labels, self.no_labels, self.CSV, self.sentences, self.labels, self.preceding_segs, self.concept1_segs, self.middle_segs, self.concept2_seg, self.suceeding_segs).data_object

        # read dataset from external files
        train_sentences = train_data['sentence']
        train_labels = train_data['label']
        print(train_labels)

        if self.test:
            test_data = SetConnection(sentences, rel_labels, no_labels)
            test_sentences = test_data['sentence_train']
            test_labels = test_data['labels_train']
>>>>>>> 8c3c10cca94e4f0182297c5a742e3ec3edb4d11e
        else:
            test_sentences = None
            test_labels = None

        self.train_label = train_labels

        if self.test:
             self.train, self.x_test, self.word_index = self.vectorize_words(train_sentences, test_sentence)
             # self.train_onehot, self.x_test_onehot, self.token_index = self.one_hot_encoding(train_sentences, test_sentence)
             self.y_test = test_labels
        else:
<<<<<<< HEAD
            # self.train_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
            self.train, self.word_index = self.vectorize_words(train_data, test_data)

        # divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
        # self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,
                                                                                                  #self.train_label)

        if segment:
            train_preceding = data_object['seg_preceding']
            train_middle = data_object['seg_middle']
            train_succeeding = data_object['seg_succeeding']
            train_concept1 = data_object['seg_concept1']
            train_concept2 = data_object['seg_concept2']
=======
             # self.train_onehot, self.token_index = self.one_hot_encoding(train_sentences, test_data)
             self.train, self.word_index = self.vectorize_words(train_sentences, test_sentences)

        # divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
        # self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,self.train_label)

        if segment:
            train_preceding = train_data['seg_preceding']
            train_middle = train_data['seg_middle']
            train_succeeding = train_data['seg_succeeding']
            train_concept1 = train_data['seg_concept1']
            train_concept2 = train_data['seg_concept2']
>>>>>>> 8c3c10cca94e4f0182297c5a742e3ec3edb4d11e

            # convert into segments
            self.preceding, self.middle, self.succeeding, self.concept1, self.concept2, self.word_index = self.vectorize_segments(
                train_data, train_preceding, train_middle, train_succeeding, train_concept1, train_concept2)

    def one_hot_encoding(self, train_list, test_list=None):
        """
        Takes a list as the input and tokenizes the samples via the `split` method.
        Assigns a unique index to each unique word and returns a dictionary of unique tokens.
        Encodes the words into one-hot vectors and stores the results in a matrix

        :param test_list: test data
        :param train_list: train data
        :return matrix: matrix with one-hot encoding
        """
        token_index = {}
        for content in train_list:
            for word in content.split():
                if word not in token_index:
                    token_index[word] = len(token_index) + 1

        # One_hot_encoding for train data
        one_hot_train = np.zeros((len(train_list), self.maxlen, max(token_index.values()) + 1))
        for i, sample in enumerate(train_list):
            for j, word in list(enumerate(sample.split()))[:self.maxlen]:
                index = token_index.get(word)
                one_hot_train[i, j, index] = 1.

        if self.test:
            # One_hot_encoding for test data
            one_hot_test = np.zeros((len(test_list), self.maxlen, max(token_index.values()) + 1))
            for i, sample in enumerate(test_list):
                for j, word in list(enumerate(sample.split()))[:self.maxlen]:
                    index = token_index.get(word)
                    one_hot_test[i, j, index] = 1.

            return one_hot_train, one_hot_test, token_index
        else:
            return one_hot_train, token_index

    def vectorize_words(self, train_list, test_list=None):

        """
        Takes training data as input (test data is optional), creates a Keras tokenizer configured to only take into account the top given number
        of the most common words in the training data and builds the word index. If test data is passed it will be tokenized using the same
        tokenizer and output the vector. If the one-hot flag is set to true, one-hot vector is returned if not vectorized sequence is returned

        :param train_list: train data
        :param test_list: test data
        :return: one-hot encoding or the vectorized sequence of the input list, unique word index
        """
        tokenizer = Tokenizer(self.common_words)

        # This builds the word index
        tokenizer.fit_on_texts(train_list)

        if self.one_hot:
            one_hot_train = tokenizer.texts_to_matrix(train_list, mode='binary')
            if self.test:
                one_hot_test = tokenizer.texts_to_matrix(test_list, mode='binary')
        else:
            # Turns strings into lists of integer indices.
            train_sequences = tokenizer.texts_to_sequences(train_list)
            padded_train = pad_sequences(train_sequences, maxlen=self.maxlen)

            if self.test:
                test_sequences = tokenizer.texts_to_sequences(test_list)
                padded_test = pad_sequences(test_sequences, maxlen=self.maxlen)

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        if self.one_hot:
            if self.test:
                return one_hot_train, one_hot_test, word_index
            else:
                return one_hot_train, word_index
        else:
            if self.test:
                return padded_train, padded_test, word_index
            else:
                return padded_train, word_index

    def vectorize_segments(self, sentences, preceding, middle, succeeding, concept1, concept2):
        """
        Takes in the sentences and segments and creates Keras tokenizer to return the vectorized segments
        :param sentences: sentences
        :param preceding: preceding segment
        :param middle: middle
        :param succeeding: succeeding
        :param concept1: concept1
        :param concept2: concept2
        :return: vectorized segments
        """
        tokenizer = Tokenizer(self.common_words)
        # This builds the word index
        tokenizer.fit_on_texts(sentences)

        preceding_sequences = tokenizer.texts_to_sequences(preceding)
        padded_preceding = pad_sequences(preceding_sequences, maxlen=self.maxlen)

        middle_sequences = tokenizer.texts_to_sequences(middle)
        padded_middle = pad_sequences(middle_sequences, maxlen=self.maxlen)

        succeeding_sequences = tokenizer.texts_to_sequences(succeeding)
        padded_succeeding = pad_sequences(succeeding_sequences, maxlen=self.maxlen)

        concept1_sequences = tokenizer.texts_to_sequences(concept1)
        padded_concept1 = pad_sequences(concept1_sequences, maxlen=self.maxlen)

        concept2_sequences = tokenizer.texts_to_sequences(concept2)
        padded_concept2 = pad_sequences(concept2_sequences, maxlen=self.maxlen)

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        return padded_preceding, padded_middle, padded_succeeding, padded_concept1, padded_concept2, word_index

    def binarize_labels(self, label_list, binarize=False):
        """
        Takes the input list and binarizes or vectorizes the labels
        If the binarize flag is set to true, it binarizes the input list in a one-vs-all fashion and outputs
        the one-hot encoding of the input list
        :param binarize: binarize flag
        :param label_list: list of text labels
        :return list:list of binarized / vectorized labels
        """
        if self.multilabel:
            self.encoder = preprocessing.MultiLabelBinarizer()
            encoder_label = self.encoder.fit_transform(label_list)
        elif self.test or binarize:
            self.encoder = preprocessing.MultiLabelBinarizer()
            encoder_label = self.encoder.fit_transform([[label] for label in label_list])
        else:
            self.encoder = preprocessing.LabelEncoder()
            encoder_label = self.encoder.fit_transform(label_list)
        return encoder_label
