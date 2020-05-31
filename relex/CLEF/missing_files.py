import os
import fnmatch

input_folder = '../../CLEF/test/ARGM/'
output_folder = '../../CLEF/test/ARGM_1/'

# label_list = ['REACTION_PRODUCT', 'STARTING_MATERIAL', 'SOLVENT', 'OTHER_COMPOUND']
label_list = ['TIME', 'TEMPERATURE', 'YIELD_OTHER', 'YIELD_PERCENT']


class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass
    for f in os.listdir(input_folder):
        if fnmatch.fnmatch(f, '*.ann'):
            annotations = {'entities': {}}
            with open(input_folder + str(f), 'r') as file:
                annotation_text = file.read()

            for line in annotation_text.split("\n"):
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                if "\t" not in line:
                    raise InvalidAnnotationError("Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                                                 % line)
                line = line.split("\t")
                if 'T' == line[0][0]:
                    tags = line[1].split(" ")
                    entity_name = tags[0]
                    entity_start = int(tags[1])
                    entity_end = int(tags[-1])
                    annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            f = open(output_folder + str(f), "a")
            temp_labels = []
            for key1, value1 in annotations['entities'].items():
                label1, start1, end1, mention1 = value1
                temp_labels.append(label1)
            # print(temp_label/s)
            check=False
            for l in temp_labels:
                if l in label_list:
                    check= True
            # check = any(item in temp_labels for item in label_list)
            if check:
                f.write(str(key1) + '\t' + str(label1) + ' ' + str(start1) + ' ' + str(
                            end1) + '\t' + str(mention1)+ '\n')
            f.close()