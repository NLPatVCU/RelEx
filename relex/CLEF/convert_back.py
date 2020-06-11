# Author : Samantha Mahendran for RelEx
import os
import sys
import fnmatch

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass

def convert(input_folder, output_folder):

    ARG1_list = ['EXAMPLE_LABEL', 'REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST', 'SOLVENT', 'OTHER_COMPOUND']
    ARGM_list = ['TIME', 'TEMPERATURE', 'YIELD_OTHER', 'YIELD_PERCENT']

    # Delete all files in the folder initially to prevent appending
    ext =".ann"
    # filelist = [f for f in os.listdir(output_folder) if f.endswith(ext)]
    # for f in filelist:
    #     os.remove(os.path.join(output_folder, f))

    for f in os.listdir(input_folder):
        if fnmatch.fnmatch(f, '*.ann'):
            print(f)
            annotations = {'entities': {}, 'relations': {}}
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
                # if 'T' == line[0][0]:
                #     tags = line[1].split(" ")
                #     entity_name = tags[0]
                #     entity_start = int(tags[1])
                #     entity_end = int(tags[-1])
                #     annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

                if 'R' == line[0][0]:  # TODO TEST THIS
                    tags = line[1].split(" ")
                    assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                    relation_name = tags[0]
                    relation_start = tags[1].split(':')[1]
                    relation_end = tags[2].split(':')[1]
                    annotations['relations'][line[0]] = (relation_name, relation_start, relation_end)

            f = open(output_folder + str(f), "a")
            # print(annotations['entities'])
            # for key in annotations['entities']:
            #     for label, start, end, context in [annotations['entities'][key]]:
            #         f.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
            for key in annotations['relations']:
                for label_rel, entity1, entity2 in [annotations['relations'][key]]:
                    if label_rel.split("-")[0] in ARG1_list:
                        label_rel = "ARG1"
                    elif label_rel.split("-")[0] in ARGM_list:
                        label_rel = "ARGM"
                    annotations['relations'][key] = (label_rel, entity1, entity2)
                    f.write(str(key) + '\t' + str(label_rel) + ' ' + 'Arg1:' + str(entity2) + ' ' + 'Arg2:' + str(entity1) + '\n')
            f.close()