# Author : Samantha Mahendran for RelEx-BERT
from sklearn import preprocessing
import BERT.preprocessing as pp
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os, logging, tempfile

class BERT_Model:

    def __init__(self, data_object, data_object_test=None, segment=False, test=False, write_Predictions = False, generalize=False, bert_path = None,common_words=10000, max_seq_length=128):
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
        :param max_seq_lengt: maximum length of the vector (default = 100)

        """

        self.write_Predictions = write_Predictions
        self.generalize = generalize
        self.segment = segment
        self.test = test
        self.common_words = common_words
        self.max_seq_length = max_seq_length
        self.data_object = data_object
        self.data_object_test = data_object_test
        self.bert_path = bert_path

        # read dataset from external files
        train_data = data_object['sentence']
        train_labels = data_object['label']
        # tracks the entity pair details for a relation
        train_track = data_object['track']

        #test files only
        if self.test:
            test_data = data_object_test['sentence']
            test_labels = data_object_test['label']
            # tracks the entity pair details for a relation
            test_track = data_object_test['track']

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
            test_labels = None

        # self.train_label = train_labels
        self.train_track = np.asarray(train_track).reshape((-1, 3))

        if self.test:
            # self.y_test = test_labels
            self.test_track = np.asarray(test_track).reshape((-1, 3))

        if self.segment:
            train_preceding = data_object['seg_preceding']
            train_middle = data_object['seg_middle']
            train_succeeding = data_object['seg_succeeding']
            train_concept1 = data_object['seg_concept1']
            train_concept2 = data_object['seg_concept2']

        # Create datasets (Only take up to max_seq_length words for memory)
        train_data = [' '.join(t.split()[0:max_seq_length]) for t in train_data]
        self.train_data = np.array(train_data, dtype=object)[:, np.newaxis]
        self.train_label = self.binarize_labels(train_labels, True).tolist()

        if self.test:
            test_data = [' '.join(t.split()[0:max_seq_length]) for t in test_data]
            self.test_data = np.array(test_data, dtype=object)[:, np.newaxis]
            self.test_label = self.binarize_labels(test_labels, True).tolist()

        print(self.train_label)
        # Instantiate tokenizer
        tokenizer = pp.create_tokenizer_from_hub_module(self.bert_path)

        # Convert data to InputExample format
        train_examples = pp.convert_text_to_input(train_text, train_label)
        test_examples = pp.convert_text_to_input(test_text, test_label)
        print(train_examples, test_examples)

        # Convert to features
        (self.train_input_ids, self.train_input_masks, self.train_segment_ids, self.train_labels
         ) = pp.convert_inputs_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
        (self.test_input_ids, self.test_input_masks, self.test_segment_ids, self.test_labels
         ) = pp.convert_inputs_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)



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
            self.encoder = preprocessing.MultiLabelBinarizer()
            encoder_label = self.encoder.fit_transform([[label] for label in label_list])
        else:
            self.encoder = preprocessing.LabelEncoder()
            encoder_label = self.encoder.fit_transform(label_list)
        return encoder_label

