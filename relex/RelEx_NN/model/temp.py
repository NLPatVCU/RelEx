import os
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import numpy as np
import evaluate

def read_from_file(file):
    """
    Reads external files and insert the content to a list. It also removes whitespace
    characters like `\n` at the end of each line

    :param file: name of the input file.
    :return : content of the file in list format
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


def get_features(text_series):
    """
    transforms text data to feature_vectors that can be used in the ml model.
    tokenizer must be available.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)


def prediction_to_label(prediction):
    tag_prob = [(labels[i], prob) for i, prob in enumerate(prediction.tolist())]
    return dict(sorted(tag_prob, key=lambda kv: kv[1], reverse=True))


train_data = read_from_file("../../../data/sentence_train")
train_labels = read_from_file("../../../data/labels_train")
df_data = pd.DataFrame(train_data, columns = ['sentence'])
# df_data.columns = ['sentence']
df_label = pd.DataFrame(train_labels, columns = ['label'])
# df_label.columns = ['label']

df_data.reset_index(drop=True, inplace=True)
df_label.reset_index(drop=True, inplace=True)
df_new = pd.concat((df_data, df_label), axis=1)
df_new.drop_duplicates(inplace=True)

df = df_new.groupby('sentence').agg({'label': lambda x: ','.join(x)})
df.reset_index(inplace=True)


df['label'] = df['label'].str.split(",")
df.columns = ['sentence', 'label']
print(df)
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df.label)
labels = multilabel_binarizer.classes_
num_classes =len(labels)
maxlen = 180
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df.sentence)
# print(df.label)
x = get_features(df.sentence)
y = multilabel_binarizer.transform(df.label)
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

filter_length = 300

model = Sequential()
model.add(Embedding(max_words, 20, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
# model.summary()

# callbacks = [
#     ReduceLROnPlateau(),
#     EarlyStopping(patience=4),
#     ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
# ]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2)
                    # callbacks=callbacks)

# cnn_model = keras.models.load_model('model-conv1d.h5')
# metrics = cnn_model.evaluate(x_test, y_test)
pred = model.predict(x_test)
print(pred)
np_pred = np.array(pred)
np_pred[np_pred<0.5]=0
np_pred[np_pred>0.5]=1
np_pred= np_pred.astype(int)
print(np_pred)
# y_pred_ind = np.argmax(pred, axis=1)
# print(y_pred_ind)
np_true = np.array(y_test)
print(np_true)
# print(x_test)

print(labels)
# for label in labels:
#     f1 = f1_score(np_true, np_pred, average='micro')
#     print(f1)


# fold_statistics = {}
# for label in labels:
#     fold_statistics[label] = {}
#     f1 = f1_score(np_true, np_pred, average='micro', labels=[label])
#     precision = precision_score(np_true, np_pred, average='micro', labels=[label])
#     recall = recall_score(np_true, np_pred, average='micro', labels=[label])
#     print(f1, precision, recall)
#     fold_statistics[label]['precision'] = precision
#     fold_statistics[label]['recall'] = recall
#     fold_statistics[label]['f1'] = f1



evaluate.evaluate_Model(np_pred, np_true, labels)
# fold_statistics = evaluate.cv_evaluation_fold(np_pred, np_true, labels)
# f1 = f1_score(np_true, np_pred, average='micro', labels=[labels])
# precision = precision_score(np_true, np_pred, average='micro', labels=[labels])
# recall = recall_score(np_true, np_pred, average='micro', labels=[labels])
# metrics = model.evaluate(x_test, y_test)
# print(metrics)
# print("{}: {}".format(model.metrics_names[0], metrics[0]))
# print("{}: {}".format(model.metrics_names[1], metrics[1]))

# y_true = [0, 1, 1, 0, 1, 0]
# y_pred = [0, 1, 1, 0, 0, 1]

# y_true = np.array([[1,0,0,0], [1,1,0,0], [1,1,1,1]])
# y_pred = np.array([[1,0,0,0], [1,1,1,0], [1,1,1,1]])
# print(f1_score(y_true, y_pred, average='micro') )