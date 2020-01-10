# RelEx

RelEx is a clinical **R**elation **E**xtraction **F**ramework to identify relations between two entities. Framework is divided into two main components: co-location based (rule based) Relation Extraction and deep learning based (CNN) Relation Extraction. 
![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

System is designed to consider two entities in a sentence and determine whether a relation exists in between the entities. RelEx includes a rule-based approach based on the co-location information of the drug entity. The co-location information of the drug determines with respect to the non-drug entity if the entity that is being referesd is a drug. The Deep learning based approach is further divided into components epending on the representation of what we feed into the Convolutional Neural Network (CNN). Deep learning based approach consists of 2 major components: Sentence-CNN  and Segment-CNN. Sentence-CNN further divides into Single label Sentence-CNN and multi label Sentence-CNN. 

## Examples

For example, the sentence
```
Once  her  hematocrit  stabilized,  she  was  started  on  a  heparin  gtt  with  coumadin overlap
```
contains a non-drug entity,gtt (Route)and two drugs Heparin and Coumadin. The non-drug entity has a relation with the closest drug occurrence Heparin but not with the other drug Coumadin.

## Installation

Create a python 3.6 virtual environment and install the requirements as given for each approach respectively.
For the rule-based approach, 
```
pip install Colocation_requirements.txt
```
For the deep learning-based approach, 
```
pip install CNN_requirements.txt
```

## Deployment

Sample dataset (some files from N2C2 2018 and i2b2 2010 corpus) and sample scripts are provided for both approaches respectively (/RelEx_Colocation/, /relex/). Sample script takes the paths for the data folder (relative path of the sample dataset) and predicts relation using the respective algorithms.

## Documentation
Both methods are separately documented. Following links directs to the respective documentations.
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
