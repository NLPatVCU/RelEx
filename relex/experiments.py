from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings

embedding_path = "../../word_embeddings/mimic3_d200.bin"
model = Model(sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg', test=False)
embedding=Embeddings(embedding_path, model)
seg_cnn = Segment_CNN(model, embedding)
