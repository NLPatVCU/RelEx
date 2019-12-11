import unittest
from RelEx_NN.model import Model
from segment import Set_Connection

class TestModel(unittest.TestCase):

    # def test_onehot_encoding(self):
        # self.assertEqual()

    # def test_vectorize_words(self):
        # self.assertEqual()

    # def test_vectorize_segments(self):
            # self.assertEqual()

    def test_binarize_labels(self):
        sample_data = Set_Connection(CSV=True, sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg' ).data_object
        model = Model(sample_data)
        sample_list = ["label1", "label2"]
        multilabel_output = model.binarize_labels(sample_list, True)
        single_label_output = model.binarize_labels(sample_list)
        self.assertEqual(multilabel_output, [[1, 0], [0, 1]])
        self.assertEqual(single_label_output. [1, 0])
