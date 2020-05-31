import configparser
import train_test
import ast
from data import Dataset
from segment import Segmentation
from CLEF import re_number, convert_back, convert_Test, convert_Train

config = configparser.ConfigParser()
config.read('configs/CLEF_pre.ini')
config.read('configs/binary_test_multi.ini')
config.read('configs/CLEF_post.ini')

# convert_Train.convert(config['CONVERSION']['train'], config['CONVERSION']['ARG1_train'], config['CONVERSION']['ARGM_train'])
# convert_Test.convert(config['CONVERSION']['test'], config['CONVERSION']['ARG1_test'], config['CONVERSION']['ARGM_test'])
# convert_Test.convert(config['CONVERSION']['dev'], config['CONVERSION']['ARG1_dev'], config['CONVERSION']['ARGM_dev'])

if config.getboolean('DEFAULT', 'no_label'):
    no_rel_label = ast.literal_eval(config.get("SEGMENTATION", "no_rel_label"))
else:
    no_rel_label = None

labels = ast.literal_eval(config.get("SEGMENTATION", "rel_labels"))
binary = config.getboolean('DEFAULT', 'binary_classification')

model_01 = ast.literal_eval(config.get("CNN_MODELS", "model_1"))
model_02 = ast.literal_eval(config.get("CNN_MODELS", "model_2"))
model_03 = ast.literal_eval(config.get("CNN_MODELS", "model_3"))

num_01 = ast.literal_eval(config.get("NUMBERING", "num_1"))
num_02 = ast.literal_eval(config.get("NUMBERING", "num_2"))
num_03 = ast.literal_eval(config.get("NUMBERING", "num_3"))

final_01 = ast.literal_eval(config.get("BACK_CONVERSION", "final_1"))
final_02 = ast.literal_eval(config.get("BACK_CONVERSION", "final_2"))
final_03 = ast.literal_eval(config.get("BACK_CONVERSION", "final_3"))

if binary:
    for label in labels[1:]:
        rel_labels = [labels[0], label]
        seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'],  config['SEGMENTATION']['test_path'],
                                              rel_labels,  no_rel_label)
        # print("glove 200 segment")
        # train_test.run_CNN_model(model_01, seg_train, seg_test)

        # print("glove 300 segment")
        # train_test.run_CNN_model(model_02, seg_train, seg_test)

        print("chem 200 segment")
        train_test.run_CNN_model(model_03, seg_train, seg_test)

else:
    seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'],config['SEGMENTATION']['test_path'],
                                             labels, no_rel_label)
    # print("glove 200 segment")
    # train_test.run_CNN_model(model_01, seg_train, seg_test)
    #
    # print("glove 300 segment")
    # train_test.run_CNN_model(model_02, seg_train, seg_test)

    print("chem 200 segment")
    train_test.run_CNN_model(model_03, seg_train, seg_test)

# re_number.append(num_01[0], num_01[1])
# re_number.append(num_02[0], num_02[1])
# re_number.append(num_03[0], num_03[1])
#
# convert_back.convert(final_01[0], final_01[1])
# convert_back.convert(final_02[0], final_02[1])
# convert_back.convert(final_03[0], final_03[1])