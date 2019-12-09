import unittest
from unittest import TestCase
from segment import Set_Connection
from RelEx_NN.model import Model
class TestPipeline(unittest.TestCase):
    def test_set_connection(self):
        data_1 = Set_Connection(CSV=False, dataset='../data/train_data', rel_labels=['problem', 'test'], no_labels=['NTeP'])
        data_2 = Set_Connection(CSV=True, sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg' )
        self.assertEqual(data_1.data_object, data_2.data_object)

    def test_one_hot_encoding(self):
        model = Model()
        one_hot_created = np.array(model.one_hot_encoding(["Hello World", "Goodbye World"])[0]).shape
        one_hot_true = np.zeros(3, 100, 4).shape
        self.assertEqual(one_hot_created, one_hot_true)
        self.assertEqual(model.one_hot_encoding(["Hello World", "Goodbye World"])[1], {"Hello":1, "World":2, "Goodbye":3})

    def test_binarize_labels(self):
        model = Model()
        self.assertEqual(model.binarize_labels(["AB", "BC"], [[1, 0], [0, 1]]))



if __name__ == '__main__':
    unittest.main()
