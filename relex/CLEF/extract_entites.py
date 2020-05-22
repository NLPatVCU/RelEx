# Author : Samantha Mahendran for RelEx

import os
import fnmatch

class InvalidAnnotationError(ValueError):
    """Raised when a given input is not in the valid format for that annotation type."""
    pass

input_folder = '../../new_data/CLEF/test/files/'
output_folder = '../../new_data/CLEF/test/ann/'
# Delete all files in the folder initially to prevent appending
ext =".ann"
filelist = [f for f in os.listdir(output_folder) if f.endswith(ext)]
for f in filelist:
    os.remove(os.path.join(output_folder, f))

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

        f = open(output_folder + str(f), "a")
        # print(annotations['entities'])
        for key in annotations['entities']:
            for label, start, end, context in [annotations['entities'][key]]:
                f.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')
        f.close()