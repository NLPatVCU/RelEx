from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from segment import Set_Connection

# Use pre-trained word embeddings
embedding_path = "../word_embeddings/glove.6B.200d.txt"

# path to sentence and label CSV files
data = Set_Connection(CSV=True, sentence_only = False, sentences='../data/sentence_train', labels='../data/labels_train',preceding_segs='../data/segments/preceding_seg', concept1_segs='../data/segments/concept1_seg',middle_segs='../data/segments/middle_seg',concept2_segs='../data/segments/concept2_seg', succeeding_segs='../data/segments/succeeding_seg' ).data_object

model = Model(data)
embedding=Embeddings(embedding_path, model)

# train CNN model
seg_cnn = Segment_CNN(model, embedding, True)
