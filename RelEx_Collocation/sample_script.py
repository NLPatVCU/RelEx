#Authour - Samantha Mahendran for RelEx_Collocation

from RelEx_Collocation.relation_extractor.relation import Relation

# sample
gold_folder = "/home/mahendrand/challenges/n2c2_2018_vcu_challenge/nlp/evaluation/n2c2_script/gold_sample"
system_folder = "/home/mahendrand/challenges/n2c2_2018_vcu_challenge/nlp/evaluation/n2c2_script/system_sample"

rel_obj = Relation(gold_folder, system_folder)
relations = rel_obj.find_relations('right')
rel_obj.write_to_file(relations)