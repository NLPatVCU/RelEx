from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
from RelEx_NN.nn import Simple_NN
from segment import Set_Connection
import sys
import os
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
    data = Set_Connection(CSV=True, sentences='../data/segments/sentence_train', labels='../data/segments/labels_train', preceding_segs='../data/segments/preceding_seg', concept1_segs='../data/segments/concept1_seg',middle_segs='../data/segments/middle_seg',concept2_segs='../data/segments/concept2_seg', succeeding_segs='../data/segments/succeeding_seg',track='../data/segments/track' ).data_object
    model = Model(data, segment=True)
    # model = Model(data, segment=True, write_Predictions=True)
    embedding=Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
    # if sys.argv[4] and sys.argv[5]:
    #     seg_cnn = Segment_CNN(model, embedding, True, final_predictions=sys.argv[5], No_Rel=sys.argv[4])
    # else:
    seg_cnn = Segment_CNN(model, embedding, True)

else:
    data = Set_Connection(CSV=True, sentence_only=True, sentences='../data/segments/sentence_train',labels='../data/segments/labels_train',track='../data/segments/track').data_object
    # data = Set_Connection(CSV=True, sentence_only = True, sentences='../data/n2c2/sentence_train', labels='../data/n2c2/labels_train').data_object
    if sys.argv[3] == 'single':
        model = Model(data)
        # model = Model(data, write_Predictions=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
        # if sys.argv[4] and sys.argv[5]:
        #     sent_cnn = Sentence_CNN(model, embedding, True, final_predictions=sys.argv[5], No_Rel=sys.argv[4])
        # else:
        sent_cnn = Sentence_CNN(model, embedding, True)
    else:
        model = Model(data, multilabel=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=int(sys.argv[2]))
        sent_cnn = Sentence_CNN(model, embedding, True, filters=300, drop_out=0.1, filter_conv=3, optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

# sent_cnn = Simple_NN(model, embedding, True)
# seg_cnn.cross_validate(model.preceding, model.middle, model.succeeding, model.concept1, model.concept2, model.train_label)
