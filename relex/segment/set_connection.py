# Author: Cora Lewis for RelEx
from data import Dataset
from segment import Segmentation
from utils import file
import os
import numpy as np
from csv import reader


class Set_Connection:
    def __init__(self, sentence_only = False, sentences=None, labels=None, preceding_segs=None, concept1_segs=None,
                 middle_segs=None, concept2_segs=None, succeeding_segs=None, track=None, dataset=None,
                 rel_labels=None, no_labels=None, CSV=True, test=False ):
        """
        Creates object based on data either from a dataset folder or a set of CSVs
        :param dataset: path to dataset
        :param rel_labels: list of options for relationship labels
        :param no_labels: list with label for when there is no relationship
        :param CSV: flag: false if creating object from dataset, true if using CSVs
        :param sentences: path to sentences CSV
        :param labels: path to labels CSV
        :param preceding_segs: path to preceding segments CSV
        :param concept1_segs: path to concept1 segments CSV
        :param middle_segs: path to middle segements CSV
        :param succeeding_segs: path to succeeding segments CSV
        """
        self.CSV = CSV
        self.sentence_only = sentence_only
        self.test = test
        if self.CSV:
            self.sentences = sentences
            self.labels = labels
            self.track = track
            if not self.sentence_only:
                self.preceding_segs = preceding_segs
                self.concept1_segs = concept1_segs
                self.middle_segs = middle_segs
                self.concept2_segs = concept2_segs
                self.succeeding_segs = succeeding_segs
            self.data_object = self.get_data_object

        else:  # if there are no CSVs, runs the Segmentation module to get object
            self.dataset = Dataset(dataset)
            self.rel_labels = rel_labels
            self.no_labels = no_labels
            self.data_object = Segmentation(self.dataset, self.rel_labels, self.no_labels, test=self.test ).segments

    @property
    def get_data_object(self):
        """
        creates object from CSVs
        """
        obj = {}

        # gets segments, labels, and sentences from file
        train_data = file.read_from_file(self.sentences)
        train_labels = file.read_from_file(self.labels)
        track_list = file.read_from_file(self.track, read_as_int=True)

        if not self.sentence_only:
            train_preceding = file.read_from_file(self.preceding_segs)
            train_concept1 = file.read_from_file(self.concept1_segs)
            train_middle = file.read_from_file(self.middle_segs)
            train_concept2 = file.read_from_file(self.concept2_segs)
            train_succeeding = file.read_from_file(self.succeeding_segs)

        # Adds segments, labels, and sentences to object
        obj['sentence'] = train_data
        obj['label'] = train_labels
        obj['track'] = track_list
        if not self.sentence_only:
            obj['seg_preceding'] = train_preceding
            obj['seg_concept1'] = train_concept1
            obj['seg_middle'] = train_middle
            obj['seg_concept2'] = train_concept2
            obj['seg_succeeding'] = train_succeeding

        return obj
