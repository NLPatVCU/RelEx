# Author : Samantha Mahendran for RelEx
import sys

import numpy
from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.model import evaluate
from RelEx_NN.embeddings.elmo_embeddings import Get_ELMo_Embeddings
from RelEx_NN.embeddings.elmo_embeddings import Convert_Tensor_Into_Numpy_Array
from RelEx_NN.embeddings.elmo_embeddings import Embed_To_File
from RelEx_NN.embeddings.elmo_embeddings import threadTime
import tensorflow as tf


class Sentence_CNN:

    def __init__(self, model, embedding, cross_validation = False, epochs=20, batch_size=512, filters=32, filter_conv=1, filter_maxPool=5,
                 activation='relu', output_activation='sigmoid', drop_out=0.5, loss='categorical_crossentropy',
                 optimizer='rmsprop', metrics=['accuracy'], elmo=None):

        self.data_model = model
        self.embedding = embedding
        self.cv = cross_validation
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.filter_conv = filter_conv
        self.filter_maxPool = filter_maxPool
        self.activation = activation
        self.output_activation = output_activation
        self.drop_out = drop_out
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.elmo = elmo
       # if self.elmo != None:

        #    sentFile = open(self.elmo)
         #   sents = sentFile.read().split("\n")
            #threadTime(sents,2)
          #  self.elmo = Convert_Tensor_Into_Numpy_Array(self.elmo)
           # sentFile.close()
        if self.cv:
            self.cross_validate()

    def define_model(self, no_classes):
        """

        :param no_classes:
        :return:
        """
        input_shape = Input(shape=(self.data_model.maxlen,))
        print(self.data_model.maxlen)
        
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)
        #print(self.embedding.embedding_dim)


        if self.embedding:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                  weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
       # if self.elmo !=None:
        #    print(type(self.elmo))
         #   self.elmo= emo.Convert_Tensor_Into_Numpy_Array(self.elmo)
            # embedding = self.elmo ##this is a tensor, not a layer
          #  print("checkpoint")



        conv1 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
        pool1 = MaxPooling1D(pool_size=self.filter_maxPool)(conv1)

        conv2 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(pool1)
        drop = Dropout(self.drop_out)(conv2)

        flat = Flatten()(drop)
        dense1 = Dense(self.filters, activation=self.activation)(flat)
        outputs = Dense(no_classes, activation=self.output_activation)(dense1)

        model = Model(inputs=input_shape, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        #print(model.summary())

        return model

    def fit_Model(self, model, x_train, y_train, validation=None):
        """

        :param model:
        :param x_train:
        :param y_train:
        :param validation:
        :return:
        """
        # print(len(x_train[1]))
        # print(y_train[0])
    
        history = model.fit(x_train, y_train, epochs=self.epochs,
                            batch_size=self.batch_size, validation_data=validation)
        print("epochs: ", self.epochs)
        loss = history.history['loss']
        acc = history.history['acc']

        if validation is not None:
            val_loss = history.history['val_loss']
            val_acc = history.history['val_acc']
            max_epoch = val_acc.index(max(val_acc)) + 1
            self.data_model.plot_graphs(loss, val_loss, 'Epochs', 'Loss', 'Training loss', 'Validation loss',
                                        'Training and validation loss')
            self.data_model.plot_graphs(acc, val_acc, 'Epochs', 'Acc', 'Training acc', 'Validation acc',
                                        'Training and validation acc')
            return model, loss, val_loss, acc, val_acc, max_epoch

        return model, loss, acc

    def cross_validate(self, num_folds=5):
        """

        :param num_folds:
        """
        X_data =  self.data_model.train
        Y_data = self.data_model.train_label

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        assert X_data is not None and Y_data is not None, \
            "Must have features and labels extracted for cross validation"

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits(X_data, Y_data)

        evaluation_statistics = {}
        fold = 1

        for train_index, test_index in skf.split(X_data, Y_data):
            binary_Y = self.data_model.binarize_labels(Y_data, True)
            x_train, x_test = X_data[train_index], X_data[test_index]
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]
            print("Training Fold %i", fold)

            labels = [str(i) for i in self.data_model.encoder.classes_]

            cv_model = self.define_model(len(self.data_model.encoder.classes_))
            cv_model, loss, acc = self.fit_Model(cv_model, x_train, y_train)

            y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)
            fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels)

            evaluation_statistics[fold] = fold_statistics
            fold += 1

        evaluate.cv_evaluation(labels, evaluation_statistics)
