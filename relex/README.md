# RelEx CNN documentation 
## Table of Contents
1. [Data Segmentation](#data_segmentation)
2. [Model](#model)
   1. [Keras Tokenizer](#k_tokenizer)
3. [Word Embeddings](#word_embeddings)
4. [Sentence CNN](#sen_cnn)
5. [Multilabel Sentence CNN](#multi_cnn)
6. [Segment Cnn](#seg_cnn)
7. [Regularization](#Regularization)

The walkthrough guide can be found [here]()

## Data segmentation <a name="data_segmentation"></a>
Text and annotation files of a dataset are filtered and passed into a dataset object which is read in along with the entity mentions of each relation category. Dataset is prepossessed to convert it to the standard format before segmentation by removing punctuation, accent marks and other diacritics that do not contribute to the context of the text, converting all letters to lowercase.

The segmentation module, identifies and extracts sentences where entities are located. Sentences are divided into the following segments and wrapped into a segmentation object:
-    preceding segment
-   concept1 segment
-   middle segment
-   concept2 segment
-   succeeding segment

When extracting sentences it checks whether the annotated relation type already exists, if not the sentences are labeled as a no-relation pair.

## Model
### Keras Tokenizer <a name="k_tokenizer"></a>

Text in each segment is converted into vector sequences using Keras tokenizer.

Keras tokenizer takes input data and creates a tokenizer. The tokenizer considers only the top given number of the most common words in the input data and creates a word index. The data is tokenized using the tokenizer and returns a vector sequence.

Maximum length of a sequence is determined and the output sequence is padded according to it. Sequences that are shorter than determined length are padded with value at the end whereas sequences longer are truncated so that they fit the desired length. Position of the padding is controlled by the arguments.

## Word embeddings<a name="word_embeddings"></a>
The word embeddings map a set of words or phrases in a vocabulary to real-valued vectors which helps to reduce the dimensionality and learn linguistic patterns in the data. Given a batch of vector sequences as input, the embedding layer converts the sequence into real-valued embedding vectors. Initially the weights are assigned randomly and gradually they are adjusted through backpropagation.

Using pre-trained word embeddings as features on CNN based methods have helped to achieve better performance in previous NLP related studies. We applied both Word2Vec and GloVe representations to train word embeddings. Word2Vec uses skip-gram and negative sampling whereas GloVe uses word-to-word co-occurrences. Word2Vec algorithm is trained over the MIMIC - III ( Medical Information Mart for Intensive Care ) and GloVe is trained over Wikipedia (2014) and Gigaword 5.

## Sentence CNN <a name="sen_cnn"></a>

Each relation consists of a pair of entities and the Sentence CNN learns the relation representation for the entire sentence as a whole. First, we identify the sentence where each relation is located and extract the sentence and we feed it into a CNN for learning.

Following figure shows the function of a single label sentence CNN.

![](https://lh6.googleusercontent.com/VzMboSkKWKdFSI3E66RiL_s0NLlLJDEGQhbEywKXEIqOnWTHm39w1vPiqy3EUr5NdxRh4q375ejzX-K-znAEifHd-UZnG517UGX11G0y7j2sBb5TD4s-SWWJ2Ptq9GqK1nEZP33c)

## Multi-label Sentence CNN <a name="multi_cnn"></a>
A sentence can have more than one distinguish mentions which eventually leads to multiple labels. We modified the sentence CNN which predicted single label for an instance to predict multiple labels for an instance.

Following figure shows when the multi-label flag is enabled the system outputs a multi-hot encoded label. Multi-label Sentence CNN is constructed different in some aspects from the single label Sentence CNN : Loss function, Choice of output layer, Multi-hot-encoding of labels.

![](https://lh5.googleusercontent.com/tdwCAwTB0fDpgockkUl8FfwIDVY6BgdExH3yOx99cX6syF00d0bmr7azeTrzSuIxZPCPnnJrQ8g39oADdmPW4J3fTdMs4VWRRecAvNR7kGXtx9wd8dt9PJYOpeXA501ujUsSTjAZ)
## Segment CNN <a name="seg_cnn"></a>
Based on where the entities are located in the sentence we can divide the sentence into segments. Different segments play different roles in determining the relation class. But each relation in Sentence-CNN is represented by an entire sentence and does not capture the positional information of the entity pairs, therefore when a sentence is divided into segments and trained by separate convolutional units.

A Sentence is explicitly segmented into 5 segments:
-   preceding - tokenized words before the first concept
-   concept 1 - tokenized words in the first concept
-   middle - tokenized words between the 2 concepts
-   concept 2 - tokenized words in the second concept
-   succeeding - tokenized words after the second concept

![](https://lh5.googleusercontent.com/_eS0O7NU9XaTM8NoO0-6ETLMF379pv25M0K22PLtni0mX5eskWrQuy196S4RA9gajiZ9zuUVIolVgO-y_iAl6hp-01jBM856rojESO1YwWIJA3oZfygQ3y5DwmdPoDdG04pMWoeD)
As the figure above shows, we construct separate convolution units for each segment and concatenate before the fixed length vector is fed to the dense layer that performs the classification.

We experiment with different sliding window sizes, filter sizes, word embeddings, loss functions to fine tune the above three models.

## Regularization

Sentence CNN and Segment CNN perform well with small filter sizes while Segment CNN performs large filter sizes. Both single and multi label Sentence CNN performed well with GloVe word embeddings whereas Segment CNN with MIMIC word embeddings.

For regularization of the model we use dropout technique on the output of convolution layer. Dropout randomly drops few nodes to prevent co-adaptation of hidden units and we set this value to 0.5 while training. We use Adam and rmsprop techniques to optimize our loss function.