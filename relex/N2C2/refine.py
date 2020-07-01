import os
import fnmatch

predictions = '../Predictions/final_predictions/'
refined_predictions = '../Predictions/refined_predictions/'

ext =".ann"
filelist = [f for f in os.listdir(refined_predictions) if f.endswith(ext)]
for f in filelist:
    os.remove(os.path.join(refined_predictions, f))


for f in os.listdir(predictions):
    print(f)
    if fnmatch.fnmatch(f, '*.ann'):
        annotations = {'entities': {}, 'relations': {}}
        with open(predictions + str(f), 'r') as file:
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

        # print(annotations['entities'])
        for key in annotations['relations']:
            # print(key, annotations['relations'][key])
            for label_rel, entity1, entity2 in [annotations['relations'][key]]:
                e1 = annotations['entities'][entity1][0]
                e2 = annotations['entities'][entity2][0]
                if e2 != label_rel.split("-")[0]:
                    # print(e2, label_rel.split("-")[0], annotations['entities'][entity2])
                    ll = list(annotations['entities'][entity2])
                    ll[0] = label_rel.split("-")[0]
                    annotations['entities'][entity2] = tuple(ll)
                    # print(e2, label_rel.split("-")[0], annotations['entities'][entity2])
                    # print("-----------------------------")

        f = open(refined_predictions + str(f), "a")
        for key in annotations['entities']:
            for label, start, end, context in [annotations['entities'][key]]:
                f.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')