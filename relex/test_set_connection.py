#set_connection tests done
import unittest
from segment import Set_Connection

class test_set_connection(unittest.TestCase):

    def test_set_connection_methods(self):
        # run the two methods
        data_method_segment = Set_Connection(CSV=False, dataset='../data/mini_dataset', rel_labels=['problem', 'test'], no_labels=['NTeP']).data_object
        data_method_CSV = Set_Connection(CSV=True, sentences='./sentence_train', labels='./labels_train', preceding_segs='./preceding_seg', concept1_segs='./concept1_seg', middle_segs='./middle_seg', concept2_segs='./concept2_seg', succeeding_segs='./succeeding_seg' ).data_object
        #strip spaces
        #the segment method has one trailing space which is why spaces need to be removed for the two methods to be compared
        data_method_segment_no_space = {key: [list.strip(' ') for list in data] for key,data in data_method_segment.items()}
        data_method_CSV_no_space = {key: [list.strip(' ') for list in data] for key,data in data_method_CSV.items()}
        # compare the two methods
        self.assertEqual(data_method_segment_no_space, data_method_CSV_no_space)

if __name__ == '__main__':
    unittest.main()
