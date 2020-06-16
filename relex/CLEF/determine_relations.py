# Author : Samantha Mahendran for RelEx

import os
import fnmatch
class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass

def delete_files(output_folder):
    ext = ".ann"
    filelist = [f for f in os.listdir(output_folder) if f.endswith(ext)]
    for f in filelist:
        os.remove(os.path.join(output_folder, f))

input_folder = '../../new_data/CLEF/test/ann/'
ARG1_folder = '../../new_data/CLEF/test/ARG1/'
ARGM_folder = '../../new_data/CLEF/test/ARGM/'

ARG1_list = ['EXAMPLE_LABEL', 'REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST', 'SOLVENT', 'OTHER_COMPOUND']
common = ['WORKUP', 'REACTION_STEP']

# Delete all files in the folder initially to prevent appending
delete_files(ARG1_folder)
delete_files(ARGM_folder)

for f in os.listdir(input_folder):
    if fnmatch.fnmatch(f, '*.ann'):
        print(f)
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

        for key in annotations['entities']:
            for label, start, end, context in [annotations['entities'][key]]:

                if label in common:
                    f1 = open(ARG1_folder + str(f), "a")
                    f1.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
                    f1.close()
                    f2 = open(ARGM_folder + str(f), "a")
                    f2.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
                    f2.close()
                elif label in ARG1_list:
                    f3 = open(ARG1_folder + str(f), "a")
                    f3.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
                    f3.close()
                else:
                    f4 = open(ARGM_folder + str(f), "a")
                    f4.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
                    f4.close()