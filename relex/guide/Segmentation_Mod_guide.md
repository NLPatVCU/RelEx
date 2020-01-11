
# Segmentation Module

Dataset is read in and the text and annotation files are segmented along with the labels.  Note that text and annotation files should be in brat format. Brat standard format can be referred here: https://brat.nlplab.org/standoff.html
## Data Segmentation

### Flags:
- no_relation : If relation doesn't exist between two entities and the flag is enabled, the entity pair is assigned to the no_relation label.
- same_entity_relation: check whether relation exists between same type of entities

### Code
Following snippet shows how to run segmentation module to extract segments of the i2b2-2010 dataset TeP (test - problem category).
*You can find a sample program [here](https://github.com/SamMahen/RelEx/tree/master/relex/samples).
```python
# To extract the problem - treatment relations object
from data import Dataset
from segment import Segmentation

# path to the dataset
sample_train = Dataset('../data/training_data')
# list of relation labels
rel_labels = ['problem', 'test']
#list of labels for when there is no relationship
no_rel_label = ['NTeP']
# extract segments from the dataset
seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label )
```
### Running Instructions
1.  Run the above code snippet (*[sample](https://github.com/SamMahen/RelEx/blob/master/relex/samples/sample_segmentation_module.py)*)
2.  Add the entities to the rel_labels list
3.  If no_relation flag is enabled, add no_relation label name to the no_rel_label list, if not do not pass the list as an argument.

If the dataset contains multiple categories of classes, run each category of the labels separately and follow the above steps.

For example, first locate the sentence where one entity is located and determine the relations between other entity types in the same sentence. When a pair of entities is identified first it checks whether an annotated relation type already exists, if it is
- yes : label it with the given annotated label
- no : if no-rel label is active label as a no_relation pair or ignore

The selected sentences are extracted into the following segments:
- Preceding - (tokenize words before the first concept)
- Concept 1 - (tokenize words in the first concept)
- Middle - (tokenize words between 2 concepts)
- Concept 2 - (tokenize words in the second concept)
- Succeeding - (tokenize words after the second concept)

## Connecting modules
To feed the extracted segments into the CNN module the Connection class is called which establishes the connection between two modules:  Connects the segmentation and CNN modules. It has 2 options:
1.  Pass the dataset and perform data segmentation and feed into the CNN module
2.  Provide path to files of the sentences, labels and the segments (Eg: CSV files)

Following snippet shows how to run connection module to feed the extracted segments or the external files into the CNN modules.

```python
from segment import SetConnection

# option 1: Extract segments from the given dataset
data = SetConnection('../data/train_data', ['problem', 'test'], ['NTeP'])

# option 2: Pass external CSV files 
data=Set_Connection(CSV=True,sentence_only = True, sentences='../data/n2c2/sentence_train', labels='../data/n2c2/labels_train').data_object
#print labels
print(data_object['label'])
```

[CNN Module](https://github.com/SamMahen/RelEx/blob/master/relex/guide/CNN_Mod_guide.md)
