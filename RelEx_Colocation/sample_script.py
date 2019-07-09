#Authour - Samantha Mahendran for RelEx_Colocation

from RelEx_Colocation.relation_extractor.relation import Relation

# Give path to the data folder and the folder needed to store the predictions
data_folder = "/home/mahendrand/challenges/n2c2_2018_vcu_challenge/nlp/evaluation/n2c2_script/gold_sample"
prediction_folder = "/home/mahendrand/challenges/n2c2_2018_vcu_challenge/nlp/evaluation/n2c2_script/system_sample"

# create object of the Relation
rel_obj = Relation(data_folder, prediction_folder)

# Give the traversal type to get the prediction. By default it takes left
"""
Different types of traversal techniques: 
    1. left - traverse only the left side of the drug mention
    2. right - traverse only the right side of the drug mention
    3. right-left - traverse right first then left of the drug mention
    4. left-right - traverse left first then right of the drug mention
    5. sentence - traverse both sides of the drug mention within the sentence boundary
"""

predicitions = rel_obj.find_relations('sentence')

#writes the predictions to the given directory
rel_obj.write_to_file(predicitions)