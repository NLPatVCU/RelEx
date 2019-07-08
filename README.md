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

### Installation

Create a python virtual environment and install the packages given in the requirements.txt

```
pip install requirements.txt
```
### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

A sample script is provided as an example. 

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Samantha Mahendran** - Main author - [SamMahen](https://github.com/SamMahen)

## License

This package is licensed under the GNU General Public License.

## Acknowledgments
- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
