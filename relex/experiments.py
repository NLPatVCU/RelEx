from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN

# embedding_path = "../word_embeddings/mimic3_d200.txt"
# model = Model(True, False)
# embedding=Embeddings(embedding_path, model)
# seg_cnn = Segment_CNN(model, embedding)
# seg_cnn.cross_validate(model.preceding, model.middle, model.succeeding, model.concept1, model.concept2, model.train_label)


model = Model(False, False)
