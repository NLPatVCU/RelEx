# Author : Samantha Mahendran for RelEx
from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.model import evaluate
from RelEx_NN.model import model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from RelEx_NN.embeddings.elmo_embeddings import ElmoEmbeddingLayer
import numpy as np
import tensorflow as tf


class Sentence_CNN:

    def __init__(self, model, embedding, sentences=None,cross_validation=False, end_to_end=False, epochs=20, batch_size=512,
                 filters=32, filter_conv=1,
                 filter_maxPool=5, activation='relu', output_activation='sigmoid', drop_out=0.5,
                 loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):

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
        self.end_to_end = end_to_end
        self.sentences = sentences
        if self.cv:
            self.cross_validate()
        elif self.end_to_end:
            self.end_to_end_test()
        else:
            self.test()

    def define_model(self, no_classes):
        """
        define a CNN model with defined parameters when the class is called
        :param no_classes: no of classes
        :return: trained model
        """
        # Define the model with different parameters and layers when running for multi label sentence CNN
        if self.data_model.multilabel:

            model = Sequential()
            model.add(Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                weights=[self.embedding.embedding_matrix], input_length=self.data_model.maxlen))
            model.add(Dropout(self.drop_out))
            model.add(Conv1D(self.filters, self.filter_conv, padding='valid', activation=self.activation, strides=1))
            model.add(GlobalMaxPool1D())
            model.add(Dense(len(self.data_model.encoder.classes_)))
            model.add(Activation(self.output_activation))

            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        else:
            elmo=False
            
            if(self.sentences!=None):
                elmo=True;

            if(elmo==False):
                input_shape = Input(shape=(self.data_model.maxlen,))
                embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

                if self.embedding:
                   embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                            weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
            else:
                input_shape = Input(shape=(1,self.data_model.maxlen),dtype=tf.string)
                embedding = ElmoEmbeddingLayer()(input_shape)

            conv1 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
            pool1 = MaxPooling1D(pool_size=self.filter_maxPool)(conv1)

            conv2 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(pool1)
            drop = Dropout(self.drop_out)(conv2)

            flat = Flatten()(drop)
            dense1 = Dense(self.filters, activation=self.activation)(flat)
            outputs = Dense(no_classes, activation=self.output_activation)(dense1)

            model = Model(inputs=input_shape, outputs=outputs)
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        print(model.summary())
        return model

    def fit_Model(self, model, x_train, y_train, validation=None):
        """
        fit the defined model to train on the data
        :param model: trained model
        :param x_train: training data
        :param y_train: training labels
        :param validation: validation data
        :return:
        """
        # Calculate the weights for each class so that we can balance the data
        # weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

        # history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation, class_weight=weights)
            
        history = model.fit(x_train, y_train, epochs=self.epochs,
                batch_size=self.batch_size, validation_data=validation)

        # loss = history.history['loss']
        # acc = history.history['acc']

        # if validation is not None:
        #     val_loss = history.history['val_loss']
        #     val_acc = history.history['val_acc']
        #     max_epoch = val_acc.index(max(val_acc)) + 1
        #     self.data_model.plot_graphs(loss, val_loss, 'Epochs', 'Loss', 'Training loss', 'Validation loss',
        #                                 'Training and validation loss')
        #     self.data_model.plot_graphs(acc, val_acc, 'Epochs', 'Acc', 'Training acc', 'Validation acc',
        #                                 'Training and validation acc')
        #     return model, loss, val_loss, acc, val_acc, max_epoch

        return model

    def test(self):
        """
        Train - Test - Split
        """

        x_train = self.data_model.train
        y_train = self.data_model.train_label
        binary_y_train = self.data_model.binarize_labels(y_train, True)

        labels = [str(i) for i in self.data_model.encoder.classes_]

        x_test = self.data_model.x_test
        y_test = self.data_model.y_test
        binary_y_test = self.data_model.binarize_labels(y_test, True)

        cv_model = self.define_model(len(self.data_model.encoder.classes_))

        if self.data_model.multilabel:
            cv_model.fit(x_train, binary_y_train, epochs=self.epochs, batch_size=self.batch_size)
            y_pred, y_true = evaluate.multilabel_predict(cv_model, x_test, binary_y_test)
        else:
            cv_model= self.fit_Model(cv_model, x_train, binary_y_train)
            y_pred, y_true = evaluate.predict(cv_model, x_test, binary_y_test, labels)
            print(confusion_matrix(y_true, y_pred))

        print(classification_report(y_true, y_pred, target_names=labels))

    def end_to_end_test(self):
        if self.data_model.multilabel:
            x_train = self.data_model.true_train_x
            y_train = self.data_model.true_train_y
            binary_y_train = self.data_model.binarize_labels(y_train, True)

            x_test = self.data_model.true_test_x
            y_test = self.data_model.true_test_y
            binary_y_test = self.data_model.binarize_labels(y_test, True)
            labels = ['no', 'yes']
        else:
            x_train = self.data_model.train
            y_train = self.data_model.train_label
            binary_y_train = self.data_model.binarize_labels(y_train, True)

            x_test = self.data_model.x_test
            y_test = self.data_model.y_test
            binary_y_test = self.data_model.binarize_labels(y_test, True)

            labels = [str(i) for i in self.data_model.encoder.classes_]
        cv_model = self.define_model(len(labels))

        cv_model = self.fit_Model(cv_model, x_train, binary_y_train)
        y_pred, y_true = evaluate.predict(cv_model, x_test, binary_y_test, labels)

        print("---------------------  binary results ---------------------------------")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=labels))

        x_train1, y_train1, x_test1, y_test1 = self.data_model.remove_instances(y_pred)
        if self.data_model.multilabel:
            self.data_model.binary_label = False
            df_train = model.reduce_duplicate_data(x_train1, y_train1)
            y_train2 = df_train.label.tolist()
            y_train1 = y_train2
            df_test = model.reduce_duplicate_data(x_test1, y_test1)
            y_test2 = df_test.label.tolist()
            y_test1 = y_test2
            train, test, word_index = self.data_model.vectorize_words(df_train.sentence, df_test.sentence)
            x_train1 = train
            x_test1 = test
        binary_y_train1 = self.data_model.binarize_labels(y_train1, True)
        labels1 = [str(i) for i in self.data_model.encoder.classes_]
        binary_y_test1 = self.data_model.binarize_labels(y_test1, True)
        cv_model1 = self.define_model(len(labels1))
        if self.data_model.multilabel:
            cv_model1.fit(x_train1, binary_y_train1, epochs=self.epochs, batch_size=self.batch_size)
            y_pred1, y_true1 = evaluate.multilabel_predict(cv_model1, x_test1, binary_y_test1)
        else:
            cv_model1= self.fit_Model(cv_model1, np.array(x_train1), np.array(binary_y_train1))
            y_pred1, y_true1 = evaluate.predict(cv_model1, np.array(x_test1), np.array(binary_y_test1), labels1)
            print("---------------------  Final results ---------------------------------")
            print(confusion_matrix(y_true1, y_pred1))
        print(classification_report(y_true1, y_pred1, target_names=labels1))

    def cross_validate(self, num_folds=5):
        """
        Train the CNN model while running cross validation.
        :param num_folds: no of fold in CV (default = 5)
        """
        exit()
        X_data = self.data_model.train
        Y_data = self.data_model.train_label
        
        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        assert X_data is not None and Y_data is not None, \
            "Must have features and labels extracted for cross validation"

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits(X_data, Y_data)
        evaluation_statistics = {}
        fold = 1

        originalclass = []
        predictedclass = []
        if self.data_model.multilabel:

            binary_Y = self.data_model.binarize_labels(Y_data, False)
            for train_index, test_index in skf.split(X_data, binary_Y.argmax(1)):
                x_train, x_test = X_data[train_index], X_data[test_index]
                y_train, y_test = binary_Y[train_index], binary_Y[test_index]
                print("Training Fold %i", fold)

                labels = [str(i) for i in self.data_model.encoder.classes_]
                cv_model = self.define_model(len(self.data_model.encoder.classes_))

                cv_model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
                y_pred, y_true = evaluate.multilabel_predict(cv_model, x_test, y_test)
                originalclass.extend(y_true)
                predictedclass.extend(y_pred)
                print("--------------------------- Results ------------------------------------")
                print(classification_report(y_true, y_pred, target_names=labels))

            print("--------------------- Results --------------------------------")
            print(classification_report(np.array(originalclass), np.array(predictedclass), target_names=labels))

        else:
            for train_index, test_index in skf.split(X_data, Y_data):
                binary_Y = self.data_model.binarize_labels(Y_data, True)
                x_train, x_test = X_data[train_index], X_data[test_index]
                y_train, y_test = binary_Y[train_index], binary_Y[test_index]
                labels = [str(i) for i in self.data_model.encoder.classes_]
                cv_model = self.define_model(len(self.data_model.encoder.classes_))
                cv_model= self.fit_Model(cv_model, x_train, y_train)
                y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)

                originalclass.extend(y_true)
                predictedclass.extend(y_pred)
                print("--------------------------- Results ------------------------------------")
                print(classification_report(y_true, y_pred, labels=labels))
                print(confusion_matrix(y_true, y_pred))
                fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels)

                evaluation_statistics[fold] = fold_statistics
                fold += 1
            print("--------------------- Results --------------------------------")
            print(classification_report(np.array(originalclass), np.array(predictedclass), target_names=labels))
            print(confusion_matrix(np.array(originalclass), np.array(predictedclass)))

            print("---------------------medacy Results --------------------------------")
            evaluate.cv_evaluation(labels, evaluation_statistics)
           
            
