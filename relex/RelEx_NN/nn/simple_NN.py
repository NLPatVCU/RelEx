# Author : Samantha Mahendran for RelEx

from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.evaluation import evaluate


class Simple_NN:

    def __init__(self, model, embedding=False, cross_validation = False, epochs=20, batch_size=512, filters=32, filter_conv=1, filter_maxPool=5,
                 activation='relu', output_activation='sigmoid', drop_out=0.5, loss='categorical_crossentropy',
                 optimizer='rmsprop', metrics=['accuracy']):

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
        if self.cv:
            self.cross_validate()

    def define_model(self, no_classes):
        """
        define a simple NN model
        :param no_classes: no of relations classes
        :return: trained model
        """
        input_shape = Input(shape=(self.data_model.train.shape[1],))
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

        #if flag is set use the pre-trained word embedding weights
        if self.embedding:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                  weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)

        flat = Flatten()(embedding)
        dense1 = Dense(self.filters, activation=self.activation)(flat)
        dense2 = Dense(self.filters, activation=self.activation)(dense1)
        outputs = Dense(no_classes, activation=self.output_activation)(dense2)

        model = Model(inputs=input_shape, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print(model.summary())

        return model

    def define_OneHot_model(self, no_classes):
        """
        define a simple NN model that takes in the input in the one-hot vector format

        :param no_classes: no of relations classes
        :return: trained model
        """
        input_shape = Input(shape=(self.data_model.train.shape[1],self.data_model.train.shape[2]))
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

        if self.embedding:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                  weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)

        flat = Flatten()(embedding)
        dense1 = Dense(self.filters, activation=self.activation)(flat)
        dense2 = Dense(self.filters, activation=self.activation)(dense1)
        outputs = Dense(no_classes, activation=self.output_activation)(dense2)

        model = Model(inputs=input_shape, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print(model.summary())

        return model

    def fit_Model(self, model, x_train, y_train, validation=None):
        """
        fit the defined model to train on the data
        :param model: defined model
        :param x_train: data
        :param y_train:labels
        :param validation: validation data
        :return: trained model, loss and accuarcy
        """
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
        Train the NN model while running cross validation.
        :param num_folds: no of CV fold (default = 5)
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
