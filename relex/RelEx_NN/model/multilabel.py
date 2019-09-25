import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
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


train_data = read_from_file("../../../data/segments/sentence_train")
train_labels = read_from_file("../../../data/segments/labels_train")
# train_data = read_from_file("../../../data/sentence_train")
# train_labels = read_from_file("../../../data/labels_train")

df_data = pd.DataFrame(train_data, columns=['sentence'])
df_label = pd.DataFrame(train_labels, columns=['label'])
df_data.reset_index(drop=True, inplace=True)
df_label.reset_index(drop=True, inplace=True)
df_new = pd.concat((df_data, df_label), axis=1)
print("total", df_new.shape[0])

duplicate_count = np.array(df_new.groupby(df_new.columns.tolist(), as_index=False).size())

duplicates =  np.sum(np.where(duplicate_count==1, 0, duplicate_count)) - sum(i > 1 for i in duplicate_count)
print("duplicates:",duplicates)
df_new.drop_duplicates(inplace=True)
df_new.reset_index(inplace = True, drop=True)

df = df_new.groupby('sentence').agg({'label': lambda x: ','.join(x)})
df.reset_index(inplace=True)
df['count'] = df['label'].str.split(",").str.len()

df_multilabel = df.copy()
df_singlelabel = df.copy()
df = df.drop(['count'], axis =1)
# print(df)
df_multilabel.drop(df_multilabel.loc[df_multilabel['count']==1].index, inplace=True)
df_multilabel = df_multilabel.drop(['count'], axis =1)
print("multi-label: ", df_multilabel.shape[0])
df_singlelabel.drop(df_singlelabel.loc[df_singlelabel['count']!=1].index, inplace=True)
df_singlelabel = df_singlelabel.drop(['count'], axis =1)
print("single-label: ",df_singlelabel.shape[0])

# df_multilabel.to_csv("multilabel.csv", index=False, header=False)
# df_singlelabel.to_csv("singlelabel.csv", index=False, header=False)
df['label'] = df['label'].str.split(",")

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df.label)
labels = multilabel_binarizer.classes_
print(labels)
num_classes = len(labels)
maxlen = 100
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df.sentence)
X_data = get_features(df.sentence)
binary_Y = multilabel_binarizer.transform(df.label)

skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X_data, binary_Y)

# fold = 1
# originalclass = []
# predictedclass = []
#
# for train_index, test_index in skf.split(X_data, binary_Y.argmax(1)):
#
#     x_train, x_test = X_data[train_index], X_data[test_index]
#     y_train, y_test = binary_Y[train_index], binary_Y[test_index]
#     print("Training Fold %i", fold)
#     print(len(x_train), len(x_test))
#     filter_length = 300
#
#     model = Sequential()
#     model.add(Embedding(max_words, 20, input_length=maxlen))
#     model.add(Dropout(0.1))
#     model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
#     model.add(GlobalMaxPool1D())
#     model.add(Dense(num_classes))
#     model.add(Activation('sigmoid'))
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
#     history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
#
#     np_pred = np.array(model.predict(x_test))
#     np_pred[np_pred < 0.5] = 0
#     np_pred[np_pred > 0.5] = 1
#     np_pred = np_pred.astype(int)
#     print(np_pred.shape)
#     np_true = np.array(y_test)
#     print(np_true.shape)
#
#     originalclass.extend(np_true)
#     predictedclass.extend(np_pred)
#     print(classification_report(np_true, np_pred,target_names=labels))
#     # print(classification_report(np_true, np_pred,target_names=labels))
#     fold += 1
# print(classification_report(np.array(originalclass), np.array(predictedclass),target_names=labels))
