# RelEx Colocation Documentation
This is a rule-based approach to extract and classify clinical relation from clinical text. It uses the colocation information of the entities to determine the relations between the entities.

## Table of Content
1.  [Installation](#installation)
2.  [Deployment](#deployment)
3.  [Algorithm](#algorithm)

## Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt

```
pip install Colocation_requirements.txt
```
## Deployment

Sample dataset (some files from N2C2 2018 corpus) and sample script are provided (/RelEx_Colocation/). The script takes the paths for the data folder (relative path of the sample dataset) and the prediction folder and predicts the relation using the method that traverses both sides of the drug entity within a sentence boundary.

## Algorithm

A breadth-first search (BFS) algorithm is used to find the closest occurrence of the drug on either side of the non-drug entity and all the closest occurrences of the drugs within a sentence.
Here are the different traversal techniques used:
- traverse left side only
- traverse right side only
- traverse left first then right
- traverse right first then left
- traversal bounded within a sentence
