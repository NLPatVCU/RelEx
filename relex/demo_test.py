from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
from segment import Set_Connection
import sys

if sys.argv[1] == 'mimic':
    if int(sys.argv[2]) == 200:
        embedding_path = "../word_embeddings/mimic3_d200.txt"
    else:
        embedding_path = "../word_embeddings/mimic3_d300.txt"
elif sys.argv[1] == 'glove':
    if int(sys.argv[2]) == 200:
        embedding_path = "../word_embeddings/glove.6B.200d.txt"
    else:
        embedding_path = "../word_embeddings/glove.6B.300d.txt"
else:
    if int(sys.argv[2]) == 200:
        embedding_path = "../word_embeddings/patent_w2v.txt"


if sys.argv[3] == 'segment':
    connection_test = Set_Connection(CSV=True, sentences='../data/test_segments/sentence_train',
                                     labels='../data/test_segments/labels_train',
                                     preceding_segs='../data/test_segments/preceding_seg',
                                     concept1_segs='../data/test_segments/concept1_seg',
                                     middle_segs='../data/test_segments/middle_seg',
                                     concept2_segs='../data/test_segments/concept2_seg',
                                     succeeding_segs='../data/test_segments/succeeding_seg',
                                     track='../data/test_segments/track').data_object

    connection_train = Set_Connection(CSV=True, sentences='../data/segments/sentence_train',
                                      labels='../data/segments/labels_train',
                                      preceding_segs='../data/segments/preceding_seg',
                                      concept1_segs='../data/segments/concept1_seg',
                                      middle_segs='../data/segments/middle_seg',
                                      concept2_segs='../data/segments/concept2_seg',
                                      succeeding_segs='../data/segments/succeeding_seg',
                                      track='../data/segments/track').data_object
    model = Model(data_object=connection_train, data_object_test=connection_test, segment=True, test=True, write_Predictions=True)
    embedding=Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
    if sys.argv[4] is not None and sys.argv[5] is not None:
        seg_cnn = Segment_CNN(model, embedding, final_predictions=sys.argv[5], write_No_rel=sys.argv[4])
    else:
        seg_cnn = Segment_CNN(model, embedding)

else:
    connection_test = Set_Connection(CSV=True, sentence_only=True, sentences='../data/test_segments/sentence_train',
                                     labels='../data/test_segments/labels_train',
                                     track='../data/test_segments/track' ).data_object

    connection_train = Set_Connection(CSV=True, sentence_only=True, sentences='../data/segments/sentence_train',
                                      labels='../data/segments/labels_train',
                                      track='../data/segments/track' ).data_object

    if sys.argv[3] == 'single':
        model = Model(data_object=connection_train, data_object_test=connection_test, test=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
        # if sys.argv[4] and sys.argv[5]:
        #     sent_cnn = Sentence_CNN(model, embedding, final_predictions=sys.argv[5], No_Rel=sys.argv[4])
        # else:
        sent_cnn = Sentence_CNN(model, embedding)
    else:
        model = Model(data_object=connection_train, data_object_test=connection_test, test=True, multilabel=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
        sent_cnn = Sentence_CNN(model, embedding, filters=300, drop_out=0.1, filter_conv=3, optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
