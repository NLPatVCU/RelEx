# Author : Samantha Mahendran for RelEx

import os
import fnmatch

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass
original_folder = '../../data/sample/test/'
input_folder = '../../data/sample/pred/'
output_folder = '../../data/medacy/output'


for f in os.listdir(input_folder):
    if fnmatch.fnmatch(f, '*.ann'):
        print(f)
        annotations = {'entities': {}, 'relations': {}}
        with open(input_folder + str(f), 'r') as file:
            annotation_text_input = file.read()

        for line in annotation_text_input.split("\n"):
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


        with open(original_folder + str(f), 'r') as file:
            annotation_text_org = file.read()

        for line in annotation_text_org.split("\n"):
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if "\t" not in line:
                raise InvalidAnnotationError("Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                                             % line)
            line = line.split("\t")
            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                annotations['relations'][line[0]] = (relation_name, relation_start, relation_end)
        print(annotations)
        # f = open(output_folder + str(f), "a")
        # # print(annotations['entities'])
        # for key in annotations['entities']:
        #     for label, start, end, context in [annotations['entities'][key]]:
        #         f.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
        # for key in annotations['relations']:
        #     # print(key, annotations['relations'][key])
        #     for label_rel, entity1, entity2 in [annotations['relations'][key]]:
        #         if label_rel == 'ARGM':
        #             e1 = annotations['entities'][entity1][0]
        #             e2 = annotations['entities'][entity2][0]
        #             new_label = e1 + "-" + e2
        #             annotations['relations'][key] = (new_label, entity1, entity2)
        #             # print(new_label)
        #             f.write(str(key) + '\t' + str(new_label) + ' ' + 'Arg1:' + str(entity1) + ' ' + 'Arg2:' + str(
        #                 entity2) + '\n')
        # f.close()