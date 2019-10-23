from data import Dataset
from segment import Segmentation

# path to the dataset
sample_train = Dataset('../data/train_data')

'''
Running instructions:
    1. Run each category of the labels separately. Comment out the rest when running
    2. Add the entities in the category to the rel_labels list
    3. To enable the no_relation, add the label name to the no_rel_label list, if not do not pass the list as an argument

'''
# To extract the problem - treatment relations object
rel_labels = ['problem', 'test']
no_rel_label = ['NTeP']

# To extract the problem - treatment relations object
# rel_labels = ['problem', 'treatment']
# no_rel_label = ['NTeP']

# To extract the problem - treatment relations object
# rel_labels = ['problem', 'test']
# no_rel_label = ['NTeP']

# to extract the segments from the dataset
seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label )

#print for testing purposes
sample_sentTrain = seg_sampleTrain.segments['seg_concept1']
print(sample_sentTrain)
