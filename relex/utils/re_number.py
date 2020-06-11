import os
import fnmatch
import pandas as pd

input_folder= '../../Predictions/Rel_Predictions/initial/MIMIC_200/'
output_folder = '../../Predictions/Rel_Predictions/final/MIMIC_200/'
# for filename in os.listdir(initial_predictions):
#     print(filename)
#     df = pd.read_csv(initial_predictions + filename, sep="\t")
#     df.columns = ['key', 'body']
#     df['key'] = df.index
#     df['key'] = 'R' + df['key'].astype(str)
#     df.to_csv(final_predictions + filename, sep='\t', index=False, header=False, mode='a')

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
            if 'T' == line[0][0]:
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                annotations['relations'][line[0]] = (relation_name, relation_start, relation_end)

    f = open(output_folder + str(f), "a")
    for key in annotations['relations']:
        # print(key, annotations['relations'][key])
        for label_rel, entity1, entity2 in [annotations['relations'][key]]:

            e1 = annotations['entities'][entity1][0]
            e2 = annotations['entities'][entity2][0]
            # dominant label comes second (to follow )
            new_label = e2 + "-" + e1
            annotations['relations'][key] = (new_label, entity1, entity2)
            # print(new_label)
            f.write(str(key) + '\t' + str(new_label) + ' ' + 'Arg1:' + str(entity2) + ' ' + 'Arg2:' + str(
                entity1) + '\n')

    f.close()