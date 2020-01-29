import pandas as pd

labels = pd.read_csv('../data/P_P/labels_train').to_numpy()
negative_label_list = ['NTeP', 'NTrP', 'NPP']
labels_binary = []

for label in labels:
    if label not in negative_label_list:
        labels_binary.append(0)
    else:
        labels_binary.append(1)

labels_binary_csv = pd.DataFrame(labels_binary).to_csv('PPBinary.csv', header=None, index=None)
