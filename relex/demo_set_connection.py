from segment import SetConnection

# method 1: get data object from dataset
data = SetConnection('../data/train_data', ['problem', 'test'], ['NTeP'])

# method 2: get data object from CSV files
# data = SetConnection(CSV=True, sentences='../data/P_P/sentence_train', labels='../data/P_P/labels_train',preceding_segs='../data/P_P/preceding_seg', concept1_segs='../data/P_P/concept1_seg',middle_segs='../data/P_P/middle_seg',concept2_segs='../data/P_P/concept2_seg', succeeding_segs='../data/P_P/succeeding_seg' )
# retrive data data
data_object = data.data_object

 #example print
print(data_object['label'])
