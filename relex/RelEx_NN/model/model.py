# Author : Samantha Mahendran for RelEx

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import os, logging, tempfile


def read_from_file(file):
    """
    Function to read external files and insert the content to a list. It also removes whitespace
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
    Function splits the input data into training and validation. By default it takes first 1000 as the validation.

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

    def __init__(self, padding=False, segment=True, test=False, common_words=10000, maxlen=100):

        self.padding = padding
        self.segment = segment
        self.test = test
        self.common_words = common_words
        self.maxlen = maxlen

        # read dataset from external files
        train_data = read_from_file("../data/sentence_train")
        train_labels = read_from_file("../data/labels_train")
        if self.test:
            test_data = read_from_file("../data/sentence_test")
            test_labels = read_from_file("../data/labels_test")
        else:
            test_data = None
            test_labels = None

        self.train_label = train_labels
        # self.train_label = self.binarize_labels(train_labels, True)
        if self.test:
            self.train, self.x_test, self.word_index = self.vectorize_words(train_data, test_data)
            self.train_onehot, self.x_test_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
            self.y_test = self.binarize_labels(test_labels)
        else:
            self.train_onehot, self.token_index = self.one_hot_encoding(train_data, test_data)
            self.train, self.word_index = self.vectorize_words(train_data, test_data)

        # divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)
        self.x_train_onehot, self.x_val_onehot, self.y_train, self.y_val = create_validation_data(self.train_onehot,
                                                                                                  self.train_label)

        if segment:
            train_preceding = read_from_file("cur/preceding_seg")
            train_middle = read_from_file("cur/middle_seg")
            train_succeeding = read_from_file("cur/succeeding_seg")
            train_concept1 = read_from_file("cur/concept1_seg")
            train_concept2 = read_from_file("cur/concept2_seg")

            self.preceding, self.middle, self.succeeding, self.concept1, self.concept2, self.word_index = self.vectorize_segments(
                train_data, train_preceding, train_middle, train_succeeding, train_concept1, train_concept2)