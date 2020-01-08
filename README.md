# RelEx

RelEx is a clinical relation extraction framework to identify the relations between two entities. It has two main components: co-location based (rule based) and deep learning based (CNN).
![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

It is designed to consider two entities in a sentence and determine whether a relation exists in between. RelEx includes a rule-based approach based on the co-location information of the drug entity. The co-location information of the drug to determine with respect to the non-drug entity if the entity is referring to the drug. Depending on the representation of what we feed into the CNN, Deep - learning based approach consists of 2 major components: Sentence-CNN  and Segment-CNN. Sentence-CNN further divides into Single label Sentence-CNN and multi label Sentence-CNN. 

### Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt
```
pip install -r requirements.txt
```
## Deployment

Sample dataset (N2C2 2018 corpus) and sample script are provided ( RelEx_Colocation/). This script takes the paths for the data folder (relative path of the sample dataset) and the prediction folder and predicts relations using the method that traverses both sides of the drug entity within a sentence boundary.
Both the data files and the predictions can be found in the sample_dataset folder (RelEx_Colocation/sample_dataset)

## Documentation
- [RelEx Co-location]()
- [RelEx_CNN]()

## Walkthrough guide
An introdcution on how to get started with RelEx
- [Guide]()


## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)
* **Cora Lewis**
* **Bridget T McInnes**


## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
