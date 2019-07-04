#Authour - Samantha Mahendran for RelEx_Collocation

"""
Converts ann file text to spacy annotation
"""
class Ann_To_Json:
    def __init__(self, annotation_text):
        self.annotations = {'entities': {}, 'relations': []}

        for line in annotation_text.split("\n"):
            if "\t" in line:
                line = line.split("\t")

                # To obtain the entities
                if 'T' in line[0]:
                    entity_id = line[0]
                    entity_span = line[-1]

                    tags = line[1].split(" ")
                    entity_name = tags[0]
                    entity_start = int(tags[1])
                    entity_end = int(tags[-1])
                    self.annotations['entities'][line[0]] = (entity_id, entity_start, entity_end, entity_name, entity_span)

                # To obtain the relations
                if 'R' in line[0]:
                    relation_id = line[0]
                    tags = line[1].split(" ")
                    relation_name = tags[0]
                    relation_start = tags[1].split(':')[1]
                    relation_end = tags[2].split(':')[1]
                    self.annotations['relations'].append((relation_id, relation_name, relation_start, relation_end))

