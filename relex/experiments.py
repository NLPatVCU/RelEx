from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings

embedding_path = "../../word_embeddings/mimic3_d200.bin"
model = Model(True, False)
embedding=Embeddings(embedding_path, model)
seg_cnn = Segment_CNN(model, embedding)
