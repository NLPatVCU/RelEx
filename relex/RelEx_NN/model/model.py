# Author : Samantha Mahendran for RelEx

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import os, logging, tempfile


def read_from_file(file):
    """
    Reads external files and insert the content to a list. It also removes whitespace
    characters like `\n` at the end of each line

    :param file: name of the input file.
    :return : content of the file in list format
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


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


def reduce_duplicate_data(train_data, train_labels):
    """
    Reads the data into one dataframe. Removes the duplicated data and merges the respective labels. Also drops the duplicates of the labels.
    :param train_data: data
    :param train_labels: labels
    """
    df_data = pd.DataFrame(train_data, columns=['sentence'])
    df_label = pd.DataFrame(train_labels, columns=['label'])
    #concatenate the dataframes
    df_data.reset_index(drop=True, inplace=True)
    df_label.reset_index(drop=True, inplace=True)
    df_new = pd.concat((df_data, df_label), axis=1)
    #drop duplicate data
    df_new.drop_duplicates(inplace=True)
    df = df_new.groupby('sentence').agg({'label': lambda x: ','.join(x)})
    df.reset_index(inplace=True)
    df['label'] = df['label'].str.split(",")
    df.columns = ['sentence', 'label']

    return df


class Model:

    def __init__(self, segment=True, test=False, multilabel=True, one_hot=False, common_words=10000, maxlen=100):
        """

        :param segment: Flag to be set to activate segment-CNN (default-True)
        :param test: Flag to be set to validate the model on the test dataset (default-False)
        :param one_hot: Flag to be set to create one-hot vectors (default-False)
        :param common_words: Number of words to consider as features (default = 10000)
        :param maxlen: maximum length of the vector (default = 100)
        """
        self.one_hot = one_hot
        self.segment = segment
        self.test = test
        self.multilabel = multilabel
        self.common_words = common_words
        self.maxlen = maxlen

        # read dataset from external files
        train_data = read_from_file("../data/segments/sentence_train")
        train_labels = read_from_file("../data/segments/labels_train")

        if self.test:
            test_data = read_from_file("../data/segments/sentence_test")
            test_labels = read_from_file("../data/segments/labels_test")
        else:
            test_data = None
            test_labels = None

        self.train_label = train_labels

        if self.multilabel:
            df_train = reduce_duplicate_data(train_data, train_labels)
            print(df_train.label)
            if self.test:
                df_test = reduce_duplicate_data(test_data, test_labels)
                self.train, self.x_test, self.word_index = self.vectorize_words(df_train.sentence, df_test.sentence)
                self.y_test = test_labels
            else:
                self.train, self.word_index = self.vectorize_words(df_train.sentence)
        else:

            if self.test:
                self.train, self.x_test, self.word_index = self.vectorize_words(train_data, test_data)
                self.train_onehot, self.x_test_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
                self.y_test = test_labels
            else:
                self.train_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
                self.train, self.word_index = self.vectorize_words(train_data, test_data)

            # divides train data into partial train and validation data
            self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
            self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,
                                                                                                      self.train_label)

        if segment:
            train_preceding = read_from_file("../data/segments/preceding_seg")
            train_middle = read_from_file("../data/segments/middle_seg")
            train_succeeding = read_from_file("../data/segments/succeeding_seg")
            train_concept1 = read_from_file("../data/segments/concept1_seg")
            train_concept2 = read_from_file("../data/segments/concept2_seg")

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

        if self.test or binarize:
            # self.encoder = preprocessing.LabelBinarizer()
            self.encoder = MultiLabelBinarizer()
        else:
            self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(label_list)
        encoder_label = self.encoder.transform(label_list)
        # no_classes = len(self.encoder.classes_)

        # bianry_encoder_label = []
        # if len(encoder_label[0]) == 1:
        #     for label in encoder_label:
        #         if label == 0:
        #             bianry_encoder_label.append([1, 0])
        #         else:
        #             bianry_encoder_label.append([0, 1])
        #
        # bianry_encoder_label = np.array(bianry_encoder_label)

        return encoder_label
        # return bianry_encoder_label