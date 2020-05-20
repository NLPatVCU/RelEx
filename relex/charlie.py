from RelEx_NN.model import Model
from segment import Set_Connection
from RelEx_NN.cnn import Sentence_CNN
from RelEx_NN.embeddings import Embeddings

data = Set_Connection(CSV=True, sentence_only = True, sentences='../data/segments/sentence_train', labels='../data/segments/labels_train').data_object
#data = Set_Connection(CSV=True, sentences='../data/segments/sentence_train', labels='../data/segments/labels_train', preceding_segs='../data/segments/preceding_seg', concept1_segs='../data/segments/concept1_seg',middle_segs='../data/segments/middle_seg',concept2_segs='../data/segments/concept2_seg', succeeding_segs='../data/segments/succeeding_seg' ).data_object
sentences = data["sentence"]
model = Model(data, segment=False, test=False, multilabel=False, one_hot=False)
embedding = Embeddings("../word_embeddings/mimic3_d200.txt", model, embedding_dim=200)
sent_cnn = Sentence_CNN(model, embedding, cross_validation=True,sentences=sentences,filters=300, drop_out=0.1, filter_conv=3, optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])


