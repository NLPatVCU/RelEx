import unittest
from segment import SetConnection

path_to_dataset = '../data/train_data'
path_to_labels = '../data/P_P/labels_train'
path_to_sentences = '../data/P_P/sentence_train'
path_to_preceding = '../data/P_P/preceding_seg'
path_to_concept1 = '../data/P_P/concept1_seg'
path_to_middle = '../data/P_P/middle_seg'
path_to_concept2 = '../data/P_P/concept2_seg'
path_to_suceeding = '../data/P_P/succeeding_seg'


class test_pipeline(unittest.TestCase):

    def test_set_connection_methods(self):
        # run the two methods
        data_method_segment = SetConnection(path_to_dataset, ['problem', 'test'], ['NTeP'])
        data_method_CSV = SetConnection(CSV=True, sentences=path_to_sentences, labels=path_to_labels, preceding_segs=path_to_preceding, concept1_segs=path_to_concept1, middle_segs=path_to_middle, concept2_segs=path_to_concept2, succeeding_segs=path_to_suceeding )

        # compare the two methods
        self.assertEqual(data_method_segment, data_method_CSV)

    # def test_segmentation(self): test segmentation correct result add later
