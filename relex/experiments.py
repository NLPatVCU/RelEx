from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN, Sentence_CNN
from segment import Set_Connection

embedding_path = "../../word_embeddings/glove.6B.200d.txt"
connection_test = Set_Connection(CSV=False, rel_labels =['problem', 'treatment'], no_labels=['NTrP'], dataset='../data/test_data' ).data_object
connection_train = Set_Connection(CSV=False, rel_labels =['problem', 'treatment'], no_labels=['NTrP'], dataset='../data/train_data' ).data_object
model = Model(data_object = connection_train, data_object_test = connection_test, test=True)
embedding=Embeddings(embedding_path, model)
seg_cnn = Sentence_CNN(model, embedding, False)
