# Author : Samantha Mahendran for RelEx
# Author : Cora Lewis for function binarize_labels

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import os, logging, tempfile


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
    Reads the data into one dataframe. Removes the duplicated data and merges the respective labels. Also drops the
    duplicates of the labels.
    :param train_data: data
    :param train_labels: labels
    :return: dataframe
    """

    df_data = pd.DataFrame(train_data, columns=['sentence'])
    df_label = pd.DataFrame(train_labels, columns=['label'])

    # concatenate the dataframes
    df_data.reset_index(drop=True, inplace=True)
    df_label.reset_index(drop=True, inplace=True)
    df_new = pd.concat((df_data, df_label), axis=1)

    # drop duplicate data
    df_new.drop_duplicates(inplace=True)
    df = df_new.groupby('sentence').agg({'label': lambda x: ','.join(x)})
    df.reset_index(inplace=True)
    df['label'] = df['label'].str.split(",")
    df.columns = ['sentence', 'label']

    return df

def convert_binary(train_labels):
    """
    # Experiment purposes for i2b2
    :param train_labels: multiclass labels
    :return: binary labels
    """
    df_label = pd.DataFrame(train_labels, columns=['label'])
    true_labels = df_label['label'].tolist()

    negative_label_list = ['NTeP', 'NTrP', 'NPP']
    labels_binary = []
    for label in true_labels:
        if label not in negative_label_list:
            labels_binary.append('yes')
        else:
            labels_binary.append('no')
    return labels_binary


class Model:

    def __init__(self, data_object, data_object_test=None, segment=False, test=False, multilabel=False, one_hot=False,
                 binary_label=False, write_Predictions = False, with_Labels = False, generalize=False, de_sample = False, common_words=10000, maxlen=100):
        """
        :param data_object: training data object
        :param data_object_test: testing data object (None -during 5 CV)
        :param segment: Flag to be set to activate segment-CNN (default-False)
        :param test: Flag to be set to validate the model on the test dataset (default-False)
        :param multilabel: Flag to be set to run sentence-CNN for multi-labels (default-False)
        :param one_hot: Flag to be set to create one-hot vectors during vectorization (default-False)
        :param binary_label: Turn labels to binary (Experiment purposes for i2b2)
        :type write_Predictions: write entities and predictions to file
        :type with_labels: Take labels of the entities into consideration
        :param generalize: flag when relations are not dependent on the first given relation label
        :param common_words: Number of words to consider as features (default = 10000)
        :param maxlen: maximum length of the vector (default = 100)

        """
        self.de_sample = de_sample
        self.write_Predictions = write_Predictions
        self.with_Labels = with_Labels
        self.generalize = generalize
        self.one_hot = one_hot
        self.segment = segment
        self.test = test
        self.multilabel = multilabel
        self.binary_label = binary_label
        self.common_words = common_words
        self.maxlen = maxlen
        self.data_object = data_object
        self.data_object_test = data_object_test

        # read dataset from external files
        train_data = data_object['sentence']
        train_labels = data_object['label']
        # tracks the entity pair details for a relation
        train_track = data_object['track']
        if self.with_Labels:
            train_concept1_label = data_object['seg_concept1_label']
            train_concept2_label = data_object['seg_concept2_label']

        # Experiment purposes for i2b2 (Not a main functionality)
        if self.binary_label:
            self.true_labels_train = train_labels
            self.true_train = train_data
            binary_labels = convert_binary(train_labels)
            train_labels = binary_labels

        #test files only
        if self.test:
            test_data = data_object_test['sentence']
            test_labels = data_object_test['label']
            # tracks the entity pair details for a relation
            test_track = data_object_test['track']
            if self.with_Labels:
                test_concept1_label = data_object['seg_concept1_label']
                test_concept2_label = data_object['seg_concept2_label']

            # Experiment purposes for i2b2 (Not a main functionality)
            if self.binary_label:
                self.true_labels_test = test_labels
                self.true_test = test_data
                binary_labels = convert_binary(test_labels)
                test_labels = binary_labels

            #to read in segments
            if segment:
                test_preceding = data_object_test['seg_preceding']
                test_middle = data_object_test['seg_middle']
                test_succeeding = data_object_test['seg_succeeding']
                test_concept1 = data_object_test['seg_concept1']
                test_concept2 = data_object_test['seg_concept2']
                self.test_preceding, self.test_middle, self.test_succeeding, self.test_concept1, self.test_concept2, self.word_index = self.vectorize_segments(
                    test_data, test_preceding, test_middle, test_succeeding, test_concept1, test_concept2)
        else:
            #when running only with train data
            test_data = None
            test_concept1_label = None
            test_concept2_label = None
            test_labels = None

        # for multilabel sentence CNN
        if self.multilabel:
            df_train = reduce_duplicate_data(train_data, train_labels)
            self.train_label = df_train.label.tolist()
            if self.test:
                df_test = reduce_duplicate_data(test_data, test_labels)
                self.train, self.x_test, self.word_index = self.vectorize_words(df_train.sentence, df_test.sentence)
                self.y_test = df_test.label.tolist()
                self.test_track = np.asarray(test_track).reshape((-1, 3))

                # Experiment purposes for i2b2 (Not a main functionality)
                if self.binary_label:
                    self.true_train_y = train_labels
                    self.true_test_y = test_labels
                    self.true_train_x, self.true_test_x, self.word_index1 = self.vectorize_words(train_data, test_data)
            else:
                self.train, self.word_index = self.vectorize_words(df_train.sentence)
        else:
            self.train_label = train_labels
            self.train_track = np.asarray(train_track).reshape((-1, 3))
            if self.with_Labels:
                self.train_concept1_label = train_concept1_label
                self.train_concept2_label = train_concept2_label

            if self.test:
                self.train, self.x_test, self.word_index = self.vectorize_words(train_data, test_data)
                self.train_onehot, self.x_test_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
                self.y_test = test_labels
                self.test_track = np.asarray(test_track).reshape((-1, 3))
                if self.with_Labels:
                    self.test_concept1_label = test_concept1_label
                    self.test_concept2_label = test_concept2_label
            else:
                self.train_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
                self.train, self.word_index = self.vectorize_words(train_data, test_data)

            # divides train data into partial train and validation data
            self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
            self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,
                                                                                                      self.train_label)
        if self.segment:
            train_preceding = data_object['seg_preceding']
            train_middle = data_object['seg_middle']
            train_succeeding = data_object['seg_succeeding']
            train_concept1 = data_object['seg_concept1']
            train_concept2 = data_object['seg_concept2']

            # convert into segments
            self.preceding, self.middle, self.succeeding, self.concept1, self.concept2, self.word_index = self.vectorize_segments(
                train_data, train_preceding, train_middle, train_succeeding, train_concept1, train_concept2)

    #function used for experiment purposes for i2b2 (Not a main functionality): in end-to-end testing
    def remove_instances(self, y_pred):
        """
        Takes a list as the input and tokenizes the samples via the `split` method.
        :param y_pred: takes the multi class predicted labels and converts them to binary
        """

        if self.segment:
            df_test = pd.DataFrame(list(
                zip(self.x_test, self.test_preceding, self.test_middle, self.test_succeeding, self.test_concept1,
                    self.test_concept2, self.true_labels_test, y_pred)),
                                   columns=['sentence', 'preceding', 'middle', 'succeeding', 'c1', 'c2', 'true',
                                            'pred'])
            df_new_test = df_test[df_test.pred != 'no']
            df_new_test = df_new_test[df_test.true != 'NTeP']
            df_train = pd.DataFrame(list(
                zip(self.train, self.preceding, self.middle, self.succeeding, self.concept1, self.concept2,
                    self.true_labels_train)),
                                    columns=['sentence', 'preceding', 'middle', 'succeeding', 'c1', 'c2', 'label'])
            df_new_train = df_train[df_train.label != 'NTeP']
            return df_new_train, df_new_test

        elif self.multilabel:
            df_test = pd.DataFrame(list(zip(self.true_test, self.true_labels_test, y_pred)),
                                   columns=['sentence', 'true', 'pred'])
            df_new_test = df_test[df_test.pred != 'no']
            df_new_test = df_new_test[df_test.true != 'NTrP']
            df_train = pd.DataFrame(list(zip(self.true_train, self.true_labels_train)), columns=['sentence', 'label'])
            df_new_train = df_train[df_train.label != 'NTrP']
            return df_new_train['sentence'].tolist(), df_new_train['label'].tolist(), df_new_test['sentence'].tolist(), \
                   df_new_test['true'].tolist()
        else:
            df_test = pd.DataFrame(list(zip(self.x_test, self.true_labels_test, y_pred)),
                                   columns=['sentence', 'true', 'pred'])
            df_new_test = df_test[df_test.pred != 'no']
            df_new_test = df_new_test[df_test.true != 'NTeP']
            df_train = pd.DataFrame(list(zip(self.train, self.true_labels_train)), columns=['sentence', 'label'])
            df_new_train = df_train[df_train.label != 'NTeP']
            return df_new_train['sentence'].tolist(), df_new_train['label'].tolist(), df_new_test['sentence'].tolist(), \
                   df_new_test['true'].tolist()

    # haven't updated for a while - does not include recent changes
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

    # cora's fix
    def binarize_labels(self, label_list, binarize=False):
        """
        Takes the input list and binarizes or vectorizes the labels
        If the binarize flag is set to true, it binarizes the input list in a one-vs-all fashion and outputs
        the one-hot encoding of the input list
        :param binarize: binarize flag
        :param label_list: list of text labels
        :return list:list of binarized / vectorized labels
        """
        if self.multilabel and not self.binary_label:
            self.encoder = MultiLabelBinarizer()
            self.encoder.fit(label_list)
            encoder_label = self.encoder.transform(label_list)

        elif self.test or binarize:
            self.encoder = preprocessing.MultiLabelBinarizer()
            encoder_label = self.encoder.fit_transform([[label] for label in label_list])
        else:
            self.encoder = preprocessing.LabelEncoder()
            encoder_label = self.encoder.fit_transform(label_list)
        return encoder_label
