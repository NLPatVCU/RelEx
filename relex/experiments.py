from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
<<<<<<< HEAD
from RelEx_NN.cnn import Segment_CNN, Sentence_CNN
from segment import Set_Connection

# embedding_path = "../../word_embeddings/mimic3_d300.txt"
# model = Model(False, False)
# embedding=Embeddings(embedding_path, model)
# sen_cnn = Sentence_CNN(model, embedding, True)


model = Model(Set_Connection(sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train', preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg', middle_segs='../data/P_P/middle_seg', concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg').data_object)
=======

embedding_path = "../../word_embeddings/mimic3_d200.bin"
model = Model(sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg', test=False)
embedding=Embeddings(embedding_path, model)
seg_cnn = Segment_CNN(model, embedding)
#sen_cnn = Sentence_Cnn(model, embedding)
>>>>>>> 8c3c10cca94e4f0182297c5a742e3ec3edb4d11e
