# RelEx CNN

Relation extraction  using a Convolutional Neural Network.
RelEx CNN is a framework that uses a CNN to extract relationships between drug entities.

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

### Algorithm
RelEx CNN has two models to be used for relationship extraction: Sentence CNN and Segment CNN. Senetence CNN uses the whole sentence in the CNN, while Segment CNN breaks down the sentence into various segem
RelEx CNN has two models to be used for relationship extraction: Sentence CNN and Segment CNN. Sentence CNN feeds the whole sentence into the CNN, while Segment CNN breaks down the sentence into various segments to feed into the CNN. Segment CNN has had better results in our experiments. 

The paths to the data sets are fed into the Model class where the data is segmented  and prepared for the CNN. The path to the word embedding is fed into the Embeddings class.  Then the outputs of Model and Embeddings is fed into one of the CNNs. ents to use in the CNN. Segment CNN has had better results in our experiments. 

#### Sentence CNN
![Sentence CNN]("sentence_cnn_1_.png")

#### Segment CNN
![Segment CNN]("segment_cnn_1_.png")


### Examples

add a sentence here later

### Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt

```
pip install requirements.txt
```
## Deployment

Add sample dataset

To prepare data for segment CNN, run demo_segment.py.  Then run experiment.py to run the model.

## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)
* **Bridget T McInnes**

## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
