import configparser
import CV, train_test
import ast
from data import Dataset
from segment import Segmentation
from CLEF import re_number, extract_entites, convert_back

config = configparser.ConfigParser()
config.read('CLEF/configs/clef_2020.ini')

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
if test:
    if binary:
        print("Please note if it is binary classification predictions must be written to files")
        extract_entites.write_entities( config['SEGMENTATION']['test_path'], config['PREDICTIONS']['final_predictions'])

        for label in rel_labels[1:]:
            rel_labels = [rel_labels[0], label]
            seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'], rel_labels, no_rel_label, config.getboolean('SEGMENTATION', 'parallelize'),
                                   config.getint('SEGMENTATION', 'no_of_cores'))

            train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                         config['CNN_MODELS']['model'], write_predictions, write_no_relations,
                         config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['binary_predictions'])

    else:
        seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'], rel_labels,no_rel_label, config.getboolean('SEGMENTATION', 'parallelize'),
                                   config.getint('SEGMENTATION', 'no_of_cores'), config['PREDICTIONS']['final_predictions'])

        train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                             config['CNN_MODELS']['model'], write_predictions, write_no_relations,
                             config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['final_predictions'])
else:
    seg_train = CV.segment(config['SEGMENTATION']['train_path'], rel_labels, no_rel_label,
                           config.getboolean('SEGMENTATION', 'parallelize'), config.getint('SEGMENTATION', 'no_of_cores'),config['PREDICTIONS']['final_predictions'])

    CV.run_CNN_model(seg_train, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                     config['CNN_MODELS']['model'], write_predictions, write_no_relations,config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['final_predictions'])

# re_number.append(config['PREDICTIONS']['binary_predictions'], config['PREDICTIONS']['final_predictions'])
# convert_back.convert(config['PREDICTIONS']['final_predictions'], config['PREDICTIONS']['converted_predictions'])