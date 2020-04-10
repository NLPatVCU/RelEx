from data import Dataset
from segment import Segmentation

# path to the dataset
sample_train = Dataset('../data/test')

'''
Running instructions: 
    1. Run each category of the labels separately. Comment out the rest when running 
    2. Add the entities in the category to the rel_labels list
    3. To enable the no_relation, add the label name to the no_rel_label list, if not do not pass the list as an argument

'''
# To extract the problem - test relations object
# rel_labels = ['problem', 'test']
# no_rel_label = ['NTeP']

# To extract the problem - treatment relations object
# rel_labels = ['problem', 'treatment']
# no_rel_label = ['NTrP']

# To extract the problem - problem relations object
rel_labels = ['problem']
no_rel_label = ['NPP']

# rel_labels = ['Chemical']
# no_rel_label = ['NoReact']

# to extract the segments from the dataset
# seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label)
seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label, same_entity_relation = True)

#print for testing purposes
# sample_sentTrain = seg_sampleTrain.segments['seg_concept1']
# print(sample_sentTrain)
