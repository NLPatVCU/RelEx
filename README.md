   # RelEx

RelEx is a clinical **Rel**ation **Ex**traction Framework to identify relations between two entities. The framework is divided into two main components: co-location based (rule based) Relation Extraction and deep learning based (CNN) Relation Extraction. 

![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

The system is designed to consider two entities in a sentence and determine whether a relation exists in between the entities. RelEx includes a rule-based approach based on the co-location information of the drug entity. The co-location information of the drug determines with respect to the non-drug entity if the entity that is being referenced is a drug. The Deep learning based approach is further divided into components depending on the representation of what we feed into the Convolutional Neural Network (CNN). The Deep learning based approach consists of 2 major components: Sentence-CNN  and Segment-CNN. Sentence-CNN further divides into single label Sentence-CNN and multi label Sentence-CNN. 

## Examples

For example, the sentence
```
Once  her  hematocrit  stabilized,  she  was  started  on  a  heparin  gtt  with  coumadin overlap
```
contains a non-drug entity,gtt (Route)and two drugs Heparin and Coumadin. The non-drug entity has a relation with the closest drug occurrence Heparin but not with the other drug Coumadin.

## Installation

Create a python 3.6 virtual environment and install the requirements as given for each respective approach.
For the rule-based approach: 
```
pip install Colocation_requirements.txt
```
For the deep learning-based approach:
```
pip install CNN_requirements.txt
```

## Deployment

Sample dataset (some files from N2C2 2018 and i2b2 2010 corpus) and sample scripts are provided for both approaches (/RelEx_Colocation/, /relex/). The sample scripts take the paths for the data folder (relative path of the sample dataset) and predicts the relation using the respective algorithms.

## Documentation
The 2 methods are documented separately. The following links directs to their respective documentations.
- [RelEx Colocation](https://github.com/SamMahen/RelEx/blob/master/RelEx_Colocation/README.md)
- [RelEx CNN](https://github.com/SamMahen/RelEx/blob/master/relex/README.md)

## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)
* **Cora Lewis**
* **Bridget T McInnes**

## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
