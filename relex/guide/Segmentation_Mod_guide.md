
# Segmentation Module

Dataset is read in and the text and annotation files are segmented along with the labels.  Note that text and annotation files should be in brat format. Brat standard format can be referred here: https://brat.nlplab.org/standoff.html

## Code
Following snippet shows how to run segmentation module to extract segments of the i2b2-2010 dataset TeP (test - problem category).
*You can find a sample program [here](https://github.com/SamMahen/RelEx/tree/master/relex/samples)*

### Flags:
No_relation : If a relation does not exist between two entities
```python
from data import Dataset
from segment import Segmentation

# path to the dataset
sample_train = Dataset('../data/training_data')
# To extract the problem - treatment relations object
rel_labels = ['problem', 'test']
no_rel_label = ['NTeP']
# to extract segments from the dataset
seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label )
```
## Running Instructions
1.  Run the above code snippet (*[sample](https://github.com/SamMahen/RelEx/blob/master/relex/samples/sample_segmentation_module.py)*)
2.  Add the entities to the rel_labels list
3.  To enable the no_relation, add the label name to the no_rel_label list, if not do not pass the list as an argument

If the dataset contains multiple categories of classes, run each category of the labels separately and follow the above steps.

The annotation object identifies the sentence where one entity is located and tries to determine the relations between other entity types in the same sentence. When a pair of entities is identified first it checks whether an annotated relation type already exists
- yes : label it with the given annotated label
- no : if no-rel label is active label as a No - relation pair or ignore

Finally it passes the sentence and the spans of the entities to the function that extracts the following segments:
- Preceding - (tokenize words before the first concept)
- Concept 1 - (tokenize words in the first concept)
- Middle - (tokenize words between 2 concepts)
- Concept 2 - (tokenize words in the second concept)
- Succeeding - (tokenize words after the second concept)

To feed the extracted segments into the CNN module Connection class is called.
Connection between two modules:  set_connection.py connects the segmentation and CNN modules. It has 2 options:
1.  Pass the dataset and perform data segmentation and feed into the CNN module
2.  Provide path to files of the sentences, labels and the segments (Eg: CSV files)
```python
from segment import SetConnection

# option 1: get data object from dataset
data = SetConnection('../data/train_data', ['problem', 'test'], ['NTeP'])

# option 2: get data object from CSV files
data=Set_Connection(CSV=True,sentence_only = True, sentences='../data/n2c2/sentence_train', labels='../data/n2c2/labels_train').data_object
#print labels
print(data_object['label'])
```

[Next Module](https://github.com/SamMahen/RelEx/blob/relex_cora/relex/guide/cnn_module.md)
