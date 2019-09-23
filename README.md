# RelEx

Relation Extractor based on the Collocation information.
![alt text](https://nlp.cs.vcu.edu/images/Edit_NanomedicineDatabase.png "Nanoinformatics")

RelEx is a rule-based framework written based on the co-location information of the drug entity. the co-location information of the drug to determine with respect to the non-drug entity if the entity is referring to the drug. 

## Algorithm

A breadth-first search (BFS) algorithm is used to find the closest occurrence of the drug on either side of the non-drug entity and all the closest occurrences of the drugs within a sentence. 
Following are the different traversal techniques used: 
- traverse left side only 
- traverse right side only
- traverse left first then right 
- traverse right first then left
- traversal bounded within a sentence

### Examples

For example, the sentence
```
Once  her  hematocrit  stabilized,  she  was  started  on  a  heparin  gtt  with  coumadinoverlap
```
contains a non-drug entity,gtt  (Route)and two drugsHeparinandCoumadinand the non-drugentity has a relation with the closest drug occurrence Heparin.

### Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt

```
pip install -r requirements.txt
```
## Deployment

Sample dataset (some files from N2C2 2018 corpus) and sample script is provided (/RelEx_Colocation/). This takes the paths for the data folder (relative path of the sample dataset) and the prediction folder and predicts relation using the method that traverses both sides of the drug entity within a sentence boundary.

## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)
* **Bridget T McInnes**

## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
