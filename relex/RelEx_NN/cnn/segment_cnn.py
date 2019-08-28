# Author : Samantha Mahendran for RelEx

from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.model import evaluate


class Segment_CNN:

    def __init__(self, model, embedding, cross_validation = False, epochs=20, batch_size=512, filters=32, filter_conv=1, filter_maxPool=5,
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

    def define_model(self):
        input_shape = Input(shape=(self.data_model.maxlen,))
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                              weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
        conv = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
        drop = Dropout(self.drop_out)(conv)
        pool = MaxPooling1D(pool_size=self.filter_maxPool)(drop)
        flat = Flatten()(pool)

        return flat, input_shape

    def build_segment_cnn(self, no_classes):

        flat1, input_shape1 = self.define_model()
        flat2, input_shape2 = self.define_model()
        flat3, input_shape3 = self.define_model()
        flat4, input_shape4 = self.define_model()
        flat5, input_shape5 = self.define_model()

        # merge
        merged = concatenate([flat1, flat2, flat3, flat4, flat5])

        # interpretation
        dense1 = Dense(18, activation=self.activation)(merged)
        outputs = Dense(no_classes, activation=self.output_activation)(dense1)

        model = Model(inputs=[input_shape1, input_shape2, input_shape3, input_shape4, input_shape5], outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # summarize
        print(model.summary())
        return model

    def cross_validate(self, num_folds=5):
    # def cross_validate(self, Pre_data, Mid_data, Suc_data, C1_data, C2_data, Y_data, num_folds=5):
        Pre_data = self.data_model.preceding
        Mid_data = self.data_model.middle
        Suc_data = self.data_model.succeeding
        C1_data = self.data_model.concept1
        C2_data = self.data_model.concept2
        Y_data = self.data_model.train_label

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits(C1_data, Y_data)
        evaluation_statistics = {}
        fold = 1

        for train_index, test_index in skf.split(C1_data, Y_data):
            binary_Y = self.data_model.binarize_labels(Y_data, True)

            pre_train, pre_test = Pre_data[train_index], Pre_data[test_index]
            mid_train, mid_test = Mid_data[train_index], Mid_data[test_index]
            suc_train, suc_test = Suc_data[train_index], Suc_data[test_index]
            c1_train, c1_test = C1_data[train_index], C1_data[test_index]
            c2_train, c2_test = C2_data[train_index], C2_data[test_index]
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]

            labels = [str(i) for i in self.data_model.encoder.classes_]

            cv_model = self.build_segment_cnn(len(self.data_model.encoder.classes_))
            cv_model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], y_train, epochs=self.epochs,
                         batch_size=self.batch_size)
            y_pred, y_true = evaluate.predict(cv_model, [pre_test, mid_test, suc_test, c1_test, c2_test], y_test,
                                              labels)
            fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels)
            evaluation_statistics[fold] = fold_statistics
            fold += 1

        evaluate.cv_evaluation(labels, evaluation_statistics)
