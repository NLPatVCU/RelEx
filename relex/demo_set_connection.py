from segment import Set_Connection

path = '../data/sample_train'

# method 1: get data object from dataset
# data = Set_Connection(dataset = path, rel_labels = ['problem', 'test'], no_labels = ['NTeP'],CSV=False)

# method 2: get data object from CSV files
data = Set_Connection(CSV=True, sentences='../data/segments/sentence_train', labels='../data/segments/labels_train',preceding_segs='../data/segments/preceding_seg', concept1_segs='../data/segments/concept1_seg',middle_segs='../data/segments/middle_seg',concept2_segs='../data/segments/concept2_seg', succeeding_segs='../data/segments/succeeding_seg' )

# retrive data data
data_object = data.data_object

 #example print
print(data_object['label'])
print(data_object['seg_concept1'])