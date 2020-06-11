# Author : Samantha Mahendran for RelEx

import os
import fnmatch
import sys

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass

input_folder = sys.argv[1]
output_folder = sys.argv[2]

for f in os.listdir(input_folder):
    if fnmatch.fnmatch(f, '*.ann'):
        print(f)
        annotations = {'relations': {}}
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
            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                annotations['relations'][line[0]] = (relation_name, relation_start, relation_end)

        f1 = open(output_folder + str(f), "a")
        for key in annotations['relations']:
            for label_rel, entity1, entity2 in [annotations['relations'][key]]:
                # if label_rel=='YIELD_OTHER-REACTION_STEP':

                # dominant label comes second (to follow )
                f1.write(str(key) + '\t' + str(label_rel) + ' ' + 'Arg1:' + str(entity1) + ' ' + 'Arg2:' + str(
                    entity2) + '\n')
        f1.close()

