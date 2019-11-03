#Authour - Samantha Mahendran for RelEx_Colocation

import os
import shutil
from operator import itemgetter

import spacy

from RelEx_Colocation.relation_extractor import traversal
from RelEx_Colocation.utils.ann_to_json import Ann_To_Json


class Relation:

    def __init__(self, data_folder, prediction_folder):

        self.data_path = data_folder
        self.prediction_path = prediction_folder

        self.ext_1 = '.txt'
        self.ext_2 = '.ann'

        # Find all files in a directory with the following extensions
        self.txt_files = [i for i in os.listdir(self.data_path) if os.path.splitext(i)[1] == self.ext_1]
        self.ann_files = [i for i in os.listdir(self.data_path) if os.path.splitext(i)[1] == self.ext_2]

        # load spacy model
        self.nlp = spacy.load('en_core_web_sm')

        # obeject to store the predictions
        self.result = {'relations': [], 'entities': []}

    def find_relations(self, traversal_direction='left'):

        file_input_path = {}

        for f in self.ann_files:
            #  get the text file name
            file = os.path.splitext(f)[0]
            t_file = file + self.ext_1
            print(f)

            with open(os.path.join(self.data_path, f)) as file_object:
                file_input_path[f] = file_object.read().strip()

            # read files and convert ann files to JSON format
            ann_to_json = Ann_To_Json(file_input_path[f])

            rel_f = os.path.join(self.data_path, t_file)

            with open(rel_f) as file_object:
                text = file_object.read()

            # use the spacy model
            doc = self.nlp(text)

            for id, start, end, label, mention in ann_to_json.annotations['entities'].values():
                #using spaCy function to find the word using the given span
                span = doc.char_span(start, end)
                self.result['entities'].append((id, label, start, end, span, f))

            #sort the entities
            sorted_entities = sorted(ann_to_json.annotations['entities'].values(), key=itemgetter(1))

            # Calls the traversal method according to the selection of the user. By default, it takes the left only direction
            # traverse left side only
            if traversal_direction == 'left':
                rel = traversal.traverse_left_only(sorted_entities, self.result, f)
            # traverse right side only
            elif traversal_direction == 'right':
                rel = traversal.traverse_right_only(sorted_entities, self.result, f)
            # traverse left side first then right side
            elif traversal_direction == 'left-right':
                rel = traversal.traverse_left_right(sorted_entities, self.result,f)
            # traverse right side then left side
            elif traversal_direction == 'right-left':
                rel = traversal.traverse_right_left(sorted_entities, self.result,f)
            #traverse both sides within the sentence boundary
            elif traversal_direction == 'sentence':
                rel = traversal.traverse_within_sentence(sorted_entities, self.result, f, doc)

        return rel

    def write_to_file(self, rel):

        # Delete all files in the folder before the prediction
        shutil.rmtree(self.prediction_path)
        os.makedirs(self.prediction_path)

        # write the relation predictions to the file
        file_relation = {}
        for label1, label2, arg1, arg2, f in rel['relations']:
            if f not in file_relation:
                file_relation[f] = []
            file_relation[f].append((label1, label2, arg1, arg2))

        # write the entity predictions to the file
        file_entity = {}
        for id, label, start, end, span, f in self.result['entities']:
            if f not in file_entity:
                file_entity[f] = []
            file_entity[f].append((id, label, start, end, str(span).replace("\n", "")))

        for file in file_relation:
            ann_rel = ""
            ann_entity = ""

            for i, rel in enumerate(file_relation[file], 1):
                label1, label2, arg1, arg2 = rel
                ann_rel += "R%i\t%s-%s Arg1:%s Arg2:%s\n" % (i, label1, label2, arg1, arg2)

            for i, ent in enumerate(file_entity[file], 1):
                id, label, start, end, span = ent
                ann_entity += "%s\t%s %i %i\t%s\n" % (id, label, start, end, span)


            output_file = os.path.join(self.prediction_path, file)
            with open(output_file, 'a') as outfile:
                outfile.write(ann_entity + ann_rel)
