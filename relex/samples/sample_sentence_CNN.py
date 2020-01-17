import sys
sys.path.append('../')

from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Sentence_CNN
from segment import Set_Connection

# Use pre-trained word embeddings
embedding_path = "../../word_embeddings/glove.6B.200d.txt"

# path to sentence and label CSV files
data = Set_Connection(CSV=True, sentence_only = True, sentences='data/sentence_train', labels='data/labels_train').data_object

model = Model(data, False)
embedding=Embeddings(embedding_path, model)

# train CNN model
sent_cnn = Sentence_CNN(model, embedding, True)