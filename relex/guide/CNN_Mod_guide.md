# CNN Module
The segmented data is fed into 3 different Convolution Neural Network (CNN) models in this module. All preprocessing needed to convert the input data into a format that can be fed into CNN is done [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/model/model.py).

## Pre-processing 
### Tokenization
Text in each segment is converted into vector sequences using Keras tokenizer. Maximum length of a sequence is determined and the output sequence is padded according to it. Sequences that are shorter than the determined length are padded with value at the end whereas sequences longer are truncated so that they fit the desired length. The position of the padding is controlled by the arguments.

```python
tokenizer = Tokenizer(self.common_words)

# This builds the word index
tokenizer.fit_on_texts(train_list)
# Turns strings into lists of integer indices.
train_sequences = tokenizer.texts_to_sequences(train_list)
padded_train = pad_sequences(train_sequences, maxlen=self.maxlen)
```
#### Flags:
*One-hot flag*: If the one-hot flag is set to true, one-hot vector is returned, if not vectorized sequence is returned. 

### Binarize labels
Takes the input list and binarizes or vectorizes the labels. If the binarize flag is set to true, it binarizes the input list in a one-vs-all fashion and outputs.
#### Flags:
-*Segment* - Flag to be set to activate segment-CNN (default-True)
-*Test* - Flag to be set to validate the model on the test dataset (default-False)
-*Multilabel* - Flag to be set to run sentence-CNN for multi-labels (default- False)
-*One_hot* - Flag to be set to create one-hot vectors (default-False)

#### Parameters:
-*common_words*: Number of words to consider as features (default = 10000)
-*max_len*: maximum length of the vector (default = 100)

### Running Instructions
For example, run the following snippet to create the input tensors for multi-label sentence CNN model.
```python
from RelEx_NN.model import Model
data = Set_Connection(CSV=True, sentence_only = True, sentences='../data/n2c2/sentence_train', labels='../data/n2c2/labels_train').data_object
model = Model(data, segment=False, test=False, multilabel=True, one_hot=False)
```
The following links directs to more guides:

-[word embeddings](https://github.com/SamMahen/RelEx/blob/master/relex/guide/word_embeddings_guide.md)

-[simple NN models](https://github.com/SamMahen/RelEx/blob/master/relex/guide/NN_guide.md)

-[CNN models](https://github.com/SamMahen/RelEx/blob/master/relex/guide/CNN_guide.md)

