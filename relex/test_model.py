import unittest
import numpy as np
from RelEx_NN.model import Model
from segment import Set_Connection

class TestModel(unittest.TestCase):

    def test_onehot_encoding(self):
        # runs model with mini dataset
        sample_data = Set_Connection(CSV=False, dataset='../data/mini_dataset', rel_labels=['problem', 'test'], no_labels=['NTeP']).data_object
        model = Model(sample_data)
        # creates onehot for sample list
        onehot = model.one_hot_encoding(['a'])
        # compares created onehot to correct
        self.assertListEqual(onehot[0].tolist(), [np.concatenate(([[0, 1]], np.zeros((99,2))), axis=0).tolist()])
        self.assertEqual(onehot[1], {'a':1})

    def test_vectorize_words(self):
        # runs model with mini dataset
        sample_data = Set_Connection(CSV=False, dataset='../data/mini_dataset', rel_labels=['problem', 'test'], no_labels=['NTeP']).data_object
        model = Model(sample_data)
        # creates vec for sample list
        vec_words = model.vectorize_words(['a'])
        # compares created vecs to correct vecs
        self.assertListEqual(vec_words[0].tolist(), [np.concatenate((np.zeros(99),[1]), axis=0).tolist()] )
        self.assertEqual(vec_words[1], {'a':1})

    def test_vectorize_segments(self):
        # runs model with mini dataset
        sample_data = Set_Connection(CSV=False, dataset='../data/mini_dataset', rel_labels=['problem', 'test'], no_labels=['NTeP']).data_object
        model = Model(sample_data)
        # creates vectors for segments
        vec_segs = model.vectorize_segments(['a b c d e'], ['a'], ['b'], ['c'], ['d'], ['e'])
        # compares created vecs and dict with correct results
        self.assertListEqual(vec_segs[0].tolist(), [np.concatenate((np.zeros(99), [1])).tolist()])
        self.assertListEqual(vec_segs[1].tolist(), [np.concatenate((np.zeros(99), [2])).tolist()])
        self.assertListEqual(vec_segs[2].tolist(), [np.concatenate((np.zeros(99), [3])).tolist()])
        self.assertListEqual(vec_segs[3].tolist(), [np.concatenate((np.zeros(99), [4])).tolist()])
        self.assertListEqual(vec_segs[4].tolist(), [np.concatenate((np.zeros(99), [5])).tolist()])
        self.assertEqual(vec_segs[5], {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

    def test_binarize_labels(self):
        # runs model with mini dataset
        sample_data = Set_Connection(CSV=False, dataset='../data/mini_dataset', rel_labels=['problem', 'test'], no_labels=['NTeP']).data_object
        model = Model(sample_data)
        # creates binarized labels with sample list
        multilabel_output = model.binarize_labels(['label1', 'label2'], True).tolist()
        single_label_output = model.binarize_labels(['label1', 'label2']).tolist()
        # compares created binarized labels to correct results
        self.assertListEqual(multilabel_output, [[1, 0], [0, 1]])
        self.assertListEqual(single_label_output, [0, 1])

if __name__ == '__main__':
    unittest.main()
