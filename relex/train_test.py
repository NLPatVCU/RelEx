from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
from segment import Set_Connection
from data import Dataset
from segment import Segmentation
import sys
import ast

def segment (train, test, labels, no_rel=None):
    print("Performing segment TEST---------------------------------------------------------------------")
    seg_test = Set_Connection(CSV=False, dataset=test, rel_labels=labels, test=True, parallelize= True, no_of_cores = 128).data_object
    print("Performing segment TRAIN---------------------------------------------------------------------")
    if no_rel:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=labels, no_labels=no_rel, parallelize= True, no_of_cores = 128).data_object
    else:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=labels, parallelize= True, no_of_cores = 128).data_object

    return seg_train, seg_test

def run_CNN_model (model,seg_train, seg_test):

    embed = model[0]
    embed_dim = int(model[1])
    cnn_model = model[2]
    no_rel = ast.literal_eval(model[3])
    predictions_path = model[4]

    if embed == 'mimic':
        if embed_dim == 200:
            embedding_path = "../word_embeddings/mimic3_d200.txt"
        else:
            embedding_path = "../word_embeddings/mimic3_d300.txt"
    elif embed == 'glove':
        if embed_dim== 200:
            embedding_path = "../word_embeddings/glove.6B.200d.txt"
        else:
            embedding_path = "../word_embeddings/glove.6B.300d.txt"
    else:
        if embed_dim == 200:
            embedding_path = "../word_embeddings/patent_w2v.txt"

    if cnn_model == 'segment':
        model = Model(data_object=seg_train, data_object_test=seg_test, segment=True, test=True,
                      write_Predictions=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        if no_rel is not None and predictions_path is not None:
            seg_cnn = Segment_CNN(model, embedding, final_predictions= predictions_path, No_Rel=no_rel)
        else:
            seg_cnn = Segment_CNN(model, embedding)