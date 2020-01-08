# RelEx Colocation Documentation

## Algorithm

A breadth-first search (BFS) algorithm is used to find the closest occurrence of the drug on either side of the non-drug entity and all the closest occurrences of the drugs within a sentence. 
Following are the different traversal techniques used: 
- traverse left side only 
- traverse right side only
- traverse left first then right 
- traverse right first then left
- traversal bounded within a sentence
