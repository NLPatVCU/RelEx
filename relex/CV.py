from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
from segment import Set_Connection
from data import Dataset
from segment import Segmentation
import sys
import ast

def segment (train, entites, no_rel=None, parallelize= False, no_of_cores = 64, predictions_folder = None):
    """
    Perform segmentation for the training and testing data
    :param no_of_cores:
    :param train: path to train data
    :param entites: list of entities that create the relations
    :param no_rel: name the label when entities that do not have relations in a sentence are considered
    :param predictions_folder: path to predictions (output) folder
    :return: segments of train and test data
    """
    if no_rel:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, no_labels=no_rel, write_Entites = True,
                                   parallelize= parallelize, no_of_cores = no_of_cores, predictions_folder = predictions_folder).data_object
    else:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, write_Entites = True,
                                   parallelize= parallelize, no_of_cores = no_of_cores, predictions_folder = predictions_folder).data_object

    return seg_train

def run_CNN_model (seg_train, embedding_path, embed_dim, cnn_model, write_predictions = False,  with_Labels = False, write_No_rel = False, initial_predictions = None, final_predictions=None):
    """
    Call CNN models
    :param cnn_model: choose the model
    :param seg_train: train segments
    :type write_Predictions: write entities and predictions to file
    :type with_labels: Take labels of the entities into consideration
    :param initial_predictions: folder to save the initial relation predictions
    :param final_predictions: folder to save the final relation predictions
    :param write_No_rel: Write the no-relation predictions back to files
    :return: None
    """
    if cnn_model == 'segment':
        model = Model(data_object=seg_train, segment=True, write_Predictions=write_predictions, with_Labels=with_Labels)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        seg_cnn = Segment_CNN(model, embedding, cross_validation = True, initial_predictions = initial_predictions,
                              final_predictions= final_predictions, write_No_rel=write_No_rel)

    elif cnn_model == 'sentence':
        model = Model(data_object=seg_train, write_Predictions=write_predictions, with_Labels=with_Labels)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        single_sent_cnn = Sentence_CNN(model, embedding, cross_validation = True, initial_predictions=initial_predictions,
                              final_predictions=final_predictions, write_No_rel=write_No_rel)
    else:
        #multilabel sentence-CNN does not have the option to write the predictions back
        model = Model(data_object=seg_train, multilabel=True)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        multi_sent_cnn = Sentence_CNN(model, embedding, cross_validation = True)