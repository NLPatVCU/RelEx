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
                 rel_labels=None, no_labels=None, CSV=True, test=False, parallelize= False, no_of_cores = 64, predictions_folder = None, write_Entites = False):
        """
        Creates data objects directly from the dataset folder and call for segmentation or take in segments (a set of CSVs)
        :type write_Entites: write entities and predictions to file
        :param sentence_only: flag when we need to consider segments for sentence CNN only
        :param sentences: path to sentences CSV
        :param labels: path to labels CSV
        :param preceding_segs: path to preceding segments CSV
        :param concept1_segs: path to concept1 segments CSV
        :param middle_segs: path to middle segements CSV
        :param succeeding_segs: path to succeeding segments CSV
        :param track: path to track information (file, first entity, second entity)
        :param dataset: path to dataset
        :param rel_labels: list of entities that create the relations
        :param no_labels: name the label when entities that do not have relations in a sentence are considered
        :param CSV: flag to decide to read from the CSVs directly
        :param test: flag to run test-segmentation options
        :param parallelize: flag to parallelize the segmentation
        :param no_of_cores: no of cores to run the parallelized segmentation
        :param predictions_folder: path to predictions (output) folder
        """
        self.CSV = CSV
        self.sentence_only = sentence_only
        self.test = test
        self.parallelize = parallelize
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

        else:
            # if there are no CSVs, runs the Segmentation module to get the segmentation object
            self.dataset = Dataset(dataset)
            self.rel_labels = rel_labels
            self.no_labels = no_labels
            if self.parallelize:
                self.data_object = Segmentation(self.dataset, self.rel_labels, self.no_labels, test=self.test, parallelize = True, no_of_cores=no_of_cores, predictions_folder = predictions_folder,write_Entites = write_Entites ).segments
            else:
                self.data_object = Segmentation(self.dataset, self.rel_labels, self.no_labels, test=self.test, predictions_folder = predictions_folder,write_Entites = write_Entites).segments

    @property
    def get_data_object(self):
        """
        creates segmentation object from CSVs
        """
        obj = {}

        # gets segments, labels, and sentences from file
        train_data = file.read_from_file(self.sentences)
        train_labels = file.read_from_file(self.labels)
        track_list = file.read_from_file(self.track)
        # track_list = file.read_from_file(self.track, read_as_int=True)

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
