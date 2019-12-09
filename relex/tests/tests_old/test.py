from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from cnn import Segment_CNN, Sentence_CNN
from segment import Set_Connection

embedding_path = "../../word_embeddings/mimic3_d300.txt"
#for sentence cnn
model = Model(Set_Connection(CSV=True, sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg').data_object, False)
embedding = Embeddings(embedding_path, model)
sen_cnn_cv = Sentence_CNN(model, embedding, True)
sen_cnn = Sentence_CNN(model, embedding, False)

#for segment CNN
model = Model(Set_Connection(CSV=True, sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg' ).data_object, True)
embedding = Embeddings(embedding_path, model)
seg_cnn_cv = Segment_CNN(model, embedding, True)
seg_cnn = Segment_CNN(model, embedding, False)

print("The stuff has been tested.")
