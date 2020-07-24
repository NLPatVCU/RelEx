import configparser
import CV, train_test
import ast
from data import Dataset
from segment import Segmentation
from N2C2 import re_number, extract_entites,refine

config = configparser.ConfigParser()
config.read('N2C2/configs/n2c2_refine.ini')

if config.getboolean('SEGMENTATION', 'no_relation'):
    no_rel_label = ast.literal_eval(config.get("SEGMENTATION", "no_rel_label"))
else:
    no_rel_label = None

test = config.getboolean('DEFAULT', 'test')
binary = config.getboolean('DEFAULT', 'binary_classification')
with_labels = config.getboolean('DEFAULT', 'with_labels')
write_predictions = config.getboolean('DEFAULT', 'write_predictions')
write_no_relations = config.getboolean('PREDICTIONS', 'write_no_relations')
rel_labels = ast.literal_eval(config.get("SEGMENTATION", "rel_labels"))

# print("Perform initial multiclass classification")
# seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'], rel_labels,no_rel_label, config.getboolean('SEGMENTATION', 'parallelize'),
#                                    config.getint('SEGMENTATION', 'no_of_cores'), config['PREDICTIONS']['multi_predictions'])
#
# train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
#                              config['CNN_MODELS']['model'], write_predictions, write_no_relations,
#                              config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['multi_predictions'])
#
#
# print("Refine NER predictions")
# refine.refine_NER(config['PREDICTIONS']['multi_predictions'],config['PREDICTIONS']['refined_predictions'])

#
print("Perform final binary classification")
extract_entites.write_entities( config['PREDICTIONS']['refined_predictions'], config['PREDICTIONS']['final_predictions'])

for label in rel_labels[1:]:
    rel_labels = [rel_labels[0], label]
    seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'],
                                             rel_labels, no_rel_label, config.getboolean('SEGMENTATION', 'parallelize'),
                                             config.getint('SEGMENTATION', 'no_of_cores'))

    train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'],
                             config.getint('CNN_MODELS', 'embedding_dim'),
                             config['CNN_MODELS']['model'], write_predictions, write_no_relations,
                             config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['binary_predictions'])

re_number.append(config['PREDICTIONS']['binary_predictions'], config['PREDICTIONS']['final_predictions'])
