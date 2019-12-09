#Author: Cora Lewis for RelEx
from data import Dataset
from segment import Segmentation
import os

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

class Set_Connection:
    def __init__(self,sentences=None, labels=None, preceding_segs=None, concept1_segs=None,
                middle_segs=None, concept2_segs=None, succeeding_segs=None, dataset=None,
                rel_labels=None, no_labels=None, CSV=True):
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
        if self.CSV:
            self.sentences = sentences
            self.labels = labels
            self.preceding_segs = preceding_segs
            self.concept1_segs = concept1_segs
            self.middle_segs = middle_segs
            self.concept2_segs = concept2_segs
            self.succeeding_segs = succeeding_segs
            self.data_object = self.get_data_object()

        else: #if there are no CSVs, runs the Segmentation module to get object
            self.dataset = Dataset(dataset)
            self.rel_labels = rel_labels
            self.no_labels = no_labels
            self.data_object = Segmentation(self.dataset, self.rel_labels, self.no_labels).segments

    def get_data_object(self):
        """
        creates object from CSVs
        """
        object = {}

        #gets segments, labels, and sentences from file
        train_data = read_from_file(self.sentences)
        train_labels = read_from_file(self.labels)
        train_preceding = read_from_file(self.preceding_segs)
        train_concept1 = read_from_file(self.concept1_segs)
        train_middle = read_from_file(self.middle_segs)
        train_concept2 = read_from_file(self.concept2_segs)
        train_succeeding = read_from_file(self.succeeding_segs)

        #Adds segments, labels, and sentences to object
        object['sentence'] = train_data
        object['label'] = train_labels
        object['seg_preceding'] = train_preceding
        object['seg_concept1'] = train_concept1
        object['seg_middle'] = train_middle
        object['seg_concept2'] = train_concept2
        object['seg_succeeding'] = train_succeeding

        return object
