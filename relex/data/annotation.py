# Inspired from MedaCy
"""
Converts ann file text to spacy annotation
"""
import os, logging

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass


class Annotation:

    def __init__(self, ann_file_path):

        if not os.path.isfile(ann_file_path):
            raise FileNotFoundError("ann_file_path is not a valid file path")
        self.annotations = {'entities': {}, 'relations': []}
        valid_IDs = ['T', 'R', 'E', 'A', 'M', 'N']
        with open(ann_file_path, 'r') as file:
            annotation_text = file.read()
        for line in annotation_text.split("\n"):
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if "\t" not in line:
                raise InvalidAnnotationError("Line chunks in ANN files are separated by tabs, see BRAT guidelines. %s"
                                             % line)
            line = line.split("\t")
            if not line[0][0] in valid_IDs:
                raise InvalidAnnotationError("Ill formated annotation file, each line must contain of the IDs: %s"
                                             % valid_IDs)
            if 'T' == line[0][0]:
                if len(line) == 2:
                    logging.warning("Incorrectly formatted entity line in ANN file (%s): %s", ann_file_path, line)
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                self.annotations['entities'][line[0]] = (entity_name, entity_start, entity_end, line[-1])

            if 'R' == line[0][0]:  # TODO TEST THIS
                tags = line[1].split(" ")
                assert len(tags) == 3, "Incorrectly formatted relation line in ANN file"
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                self.annotations['relations'].append((relation_name, relation_start, relation_end))
            if 'E' == line[0][0]:
                logging.warning("Event annotations not implemented in medaCy")
            if 'A' == line[0][0] or 'M' == line[0][0]:
                logging.warning("Attribute annotations not implemented in medaCy")
            if 'N' == line[0][0]:
                raise NotImplementedError("Normalization annotations are not implemented in medaCy")
