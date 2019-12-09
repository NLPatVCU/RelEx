import os
import pandas as pd
import numpy as np

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


train_data = read_from_file("../../../data/segments/sentence_train")
train_labels = read_from_file("../../../data/segments/labels_train")


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
np.savetxt(r'm_label/sentence_train.txt', df_multilabel.sentence, fmt='%s')
np.savetxt(r'm_label/labels_train.txt', df_multilabel.label, fmt='%s')
np.savetxt(r's_label/sentence_train.txt', df_singlelabel.sentence, fmt='%s')
np.savetxt(r's_label/labels_train.txt', df_singlelabel.label, fmt='%s')
np.savetxt(r'total/sentence_train.txt', df.sentence, fmt='%s')
np.savetxt(r'total/labels_train.txt', df.label, fmt='%s')

# df_multilabel.to_csv("multilabel.csv", index=False, header=False)
<<<<<<< HEAD
# df_singlelabel.to_csv("singlelabel.csv", index=False, header=False)
=======
# df_singlelabel.to_csv("singlelabel.csv", index=False, header=False)
>>>>>>> relex_Sam
