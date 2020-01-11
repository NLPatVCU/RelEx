from data import Dataset
from segment import Segmentation

# path to the dataset
sample_train = Dataset('../data/training_data')
# To extract the problem - treatment relations object
rel_labels = ['problem', 'test']
no_rel_label = ['NTeP']
# to extract segments from the dataset
seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label )