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
#flag to set cross validation. If set to true it will run 5 CV or train-test split
cv = False
write_file=True
embedding_path = "../../../../word_embeddings/mimic3_d300.txt"
output_txt_path = "/home/cora/Desktop/output/output.txt"
output_csv_path = "/home/cora/Desktop/output/output.csv"

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

def read_embeddings_from_file(path):
    """
    Function to read external embedding files to build an index mapping words (as strings)
    to their vector representation (as number vectors).
    :return dictionary: word vectors
    """
    print("Reading external embedding file ......")
    if not os.path.isfile(path):
        raise FileNotFoundError("Not a valid file path")

    embeddings_index = {}
    with open(path) as f:
        next(f)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    return embeddings_index

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

def output_to_file(true, pred, target):
    """
    Function to create .txt file and csv file of classification report
    """
    report = classification_report(true, pred, target_names=target)
    report_dict = classification_report(true, pred, target_names=target, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    #writes .txt file with results
    txt_file = open(output_txt_path, 'a')
    txt_file.write(report)
    txt_file.close()

    # writes csv file
    csv_report = df_report.to_csv()
    csv_file = open(output_csv_path, 'a')
    csv_file.write(csv_report)
    csv_file.close()



embeddings_index = read_embeddings_from_file(embedding_path)
embedding_dim = 300
maxlen = 300
max_words = 5000


embeddings_index = {}
with open(embedding_path) as f:
    next(f)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

train_data = read_from_file("../../../data/P_Te/sentence_train")
train_labels = read_from_file("../../../data/P_Te/labels_train")
# train_data = read_from_file("../../../data/sentence_train")
# train_labels = read_from_file("../../../data/labels_train")

df_data = pd.DataFrame(train_data, columns=['sentence'])
df_label = pd.DataFrame(train_labels, columns=['label'])
df_data.reset_index(drop=True, inplace=True)
df_label.reset_index(drop=True, inplace=True)
df_new = pd.concat((df_data, df_label), axis=1)

df_new.drop_duplicates(inplace=True)
df_new.reset_index(inplace = True, drop=True)

df = df_new.groupby('sentence').agg({'label': lambda x: ','.join(x)})

df.reset_index(inplace=True)
df['label'] = df['label'].str.split(",")


# df = Connection(True, False, None, "../data/P_Te/sentence_train", "../data/P_Te/labels_train" )

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['label'])
labels = multilabel_binarizer.classes_
print(labels)
num_classes = len(labels)
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['sentence'])
X_data = get_features(df['sentence'])
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
binary_Y = multilabel_binarizer.transform(df['label'])

if cv:
    #cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X_data, binary_Y)

    fold = 1
    originalclass = []
    predictedclass = []

    for train_index, test_index in skf.split(X_data, binary_Y.argmax(1)):

        x_train, x_test = X_data[train_index], X_data[test_index]
        y_train, y_test = binary_Y[train_index], binary_Y[test_index]
        print("Training Fold %i" % fold)
        print(len(x_train), len(x_test))
        filter_length = 500

        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen))
        model.add(Dropout(0.1))
        model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPool1D())
        model.add(Dense(num_classes))
        model.add(Activation('sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
        history = model.fit(x_train, y_train, epochs=20, batch_size=32)

        np_pred = np.array(model.predict(x_test))
        np_pred[np_pred < 0.5] = 0
        np_pred[np_pred > 0.5] = 1
        np_pred = np_pred.astype(int)
        np_true = np.array(y_test)

        originalclass.extend(np_true)
        predictedclass.extend(np_pred)
        print(classification_report(np_true, np_pred,target_names=labels))
        # print(classification_report(np_true, np_pred,target_names=labels))
        fold += 1
    print(classification_report(np.array(originalclass), np.array(predictedclass),target_names=labels))
    if write_file:
        output_to_file()
else:
    # train - test split

    x_train, x_test, y_train, y_test = train_test_split(X_data, binary_Y, test_size=0.2, random_state=9000)
    filter_length = 300

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen))
    model.add(Dropout(0.1))
    model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(len(labels)))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=16,
                        validation_split=0.1)
    metrics = model.evaluate(x_test, y_test)
    print("{}: {}".format(model.metrics_names[0], metrics[0]))
    print("{}: {}".format(model.metrics_names[1], metrics[1]))
    np_pred = np.array(model.predict(x_test))

    np_pred[np_pred < 0.5] = 0
    np_pred[np_pred > 0.5] = 1
    np_pred = np_pred.astype(int)
    np_true = np.array(y_test)

    print(classification_report(np_true, np_pred, target_names=labels))
    if write_file:
        output_to_file(np_true, np_pred, labels)
