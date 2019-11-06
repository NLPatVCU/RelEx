# RelEx CNN

Relation Extractor using a Convolutional Neural Network.
![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

RelEx CNN is a framework that uses a CNN to extract relationships between drug entities.

## Algorithm

### Sentence CNN
![sentence CNN]("sentence_cnn_1_.png")

### Segment CNN
![segment CNN]("segment_cnn_1_.png")


### Examples

For example, the sentence
```
Once  her  hematocrit  stabilized,  she  was  started  on  a  heparin  gtt  with  coumadinoverlap
```
contains a non-drug entity,gtt  (Route)and two drugsHeparinandCoumadinand the non-drugentity has a relation with the closest drug occurrence Heparin.

### Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt

```
pip install requirements.txt
```
## Deployment

Sample dataset (some files from N2C2 2018 corpus) and sample script is provided (/RelEx/relex). This takes the paths for the data folder (relative path of the sample dataset) and the prediction folder and predicts relation using the method that traverses both sides of the drug entity within a sentence boundary.

## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)
* **Bridget T McInnes**

## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
