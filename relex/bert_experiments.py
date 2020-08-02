# Author : Samantha Mahendran for RelEx-BERT

from BERT import BERT_Model
from BERT import Run_BERT
from segment import Set_Connection
import sys

if sys.argv[1] == 'segment':
    data = Set_Connection(CSV=True, sentences='../data/segments/sentence_train', labels='../data/segments/labels_train', preceding_segs='../../data/segments/preceding_seg', concept1_segs='../../data/segments/concept1_seg',middle_segs='../../data/segments/middle_seg',concept2_segs='../../data/segments/concept2_seg', succeeding_segs='../data/segments/succeeding_seg',track='../../data/segments/track' ).data_object
    model = BERT_Model(data, segment=True, bert_path=sys.argv[2])

else:
    data = Set_Connection(CSV=True, sentence_only=True, sentences='../data/segments/sentence_train',labels='../data/segments/labels_train',track='../data/segments/track').data_object
    model = BERT_Model(data, bert_path=sys.argv[2])
    Run_BERT(model)


