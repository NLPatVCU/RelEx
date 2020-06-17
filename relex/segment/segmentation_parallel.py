# Author : Samantha Mahendran for RelEx

from data import Annotation
from utils import file, normalization
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os


def add_file_segments(doc_segments, segment):
    """
    Function to add the local segment object to the global segment object
    :param doc_segments: global segment object
    :param segment: local segment object
    :return: doc_segments
    """
    doc_segments['preceding'].extend(segment['preceding'])
    doc_segments['concept1'].extend(segment['concept1'])
    doc_segments['middle'].extend(segment['middle'])
    doc_segments['concept2'].extend(segment['concept2'])
    doc_segments['succeeding'].extend(segment['succeeding'])
    doc_segments['sentence'].extend(segment['sentence'])
    doc_segments['label'].extend(segment['label'])
    doc_segments['track'].extend(segment['track'])

    return doc_segments


def extract_Segments(sentence, span1, span2):
    """
    Takes a sentence and the span of both entities as the input. First it locates the entities in the sentence and
    divides the sentence into following segments:

    Preceding - (tokenized words before the first concept)
    concept 1 - (tokenized words in the first concept)
    Middle - (tokenized words between 2 concepts)
    concept 2 - (tokenized words in the second concept)
    Succeeding - (tokenized words after the second concept)

    :param sentence: the sentence where both entities exist
    :param span1: span of the first entity
    :param span2: span of the second entity
    :return: preceding, middle, succeeding
    """

    preceding = sentence[0:sentence.find(span1)]
    preceding = normalization.remove_Punctuation(str(preceding)).strip()

    middle = sentence[sentence.find(span1) + len(span1):sentence.find(span2)]
    middle = normalization.remove_Punctuation(str(middle)).strip()

    succeeding = sentence[sentence.find(span2) + len(span2):]
    succeeding = normalization.remove_Punctuation(str(succeeding)).strip()

    return preceding, middle, succeeding


def write_entities_to_file(ann, file, output_folder):
    """
    write the entities of a file to the prediction folder where the relations will be added later
    :param ann: annotations
    :param output_folder: folder where the predicted files are stored for evaluation
    """
    f = open(output_folder + file, "a")
    # print(annotations['entities'])
    for key in ann.annotations['entities']:
        for label, start, end, context in [ann.annotations['entities'][key]]:
            f.write(str(key) + '\t' + str(label) + ' ' + str(start) + ' ' + str(end) + '\t' + str(context) + '\n')


class Segmentation:

    def __init__(self, dataset=None, rel_labels=None, no_rel_label=None, sentence_align=False, test=False,
                 same_entity_relation=False, generalize=False, write_Predictions = False, prediction_folder = None, de_sample=None):

        """
           Dataset is read in and the text and annotation files are segmented along with the labels.
           :param dataset: path to dataset
           :param prediction_folder: path to predictions folder
           :param rel_labels: list of options for relationship labels
           :param no_labels: list with label for when there is no relationship
           :param sentence_align: options to break sentences
           :param sentences: path to sentences CSV
           :param test: test data present
           :param same_entity_relation: check whether relation exists between same type of entities
           :param de_sample: reduce the no of samples
           :param generalize: relations not dependent on the first given relation label
           :param write_Predictions: write entities and predictions to file

        """
        self.write_Predictions = write_Predictions
        self.prediction_folder = prediction_folder
        self.dataset = dataset
        self.rel_labels = rel_labels
        self.test = test
        self.same_entity_relation = same_entity_relation
        self.generalize = generalize
        self.nlp_model = English()
        if no_rel_label:
            self.no_rel_label = no_rel_label
        else:
            self.no_rel_label = False

        if de_sample:
            self.de_sample = de_sample
        else:
            self.de_sample = False

        if sentence_align:
            sentencizer = Sentencizer(punct_chars=["\n"])
        else:
            sentencizer = Sentencizer(punct_chars=["\n", ".", "?"])

        if self.write_Predictions and prediction_folder is not None:
            ext = ".ann"
            file.delete_all_files(prediction_folder, ext)

        self.nlp_model.add_pipe(sentencizer)

        # self.nlp_model = spacy.load('en_core_web_sm')

        # global segmentation object that returns all segments and the label
        self.segments = {'seg_preceding': [], 'seg_concept1': [], 'seg_concept2': [], 'seg_middle': [],
                         'seg_succeeding': [], 'sentence': [], 'label': [], 'track':[]}
        #
        # segments_file = Parallel(n_jobs=8, prefer="threads")(
        #     delayed(self.process_file)(datafile, txt_path, ann_path) for (datafile, txt_path, ann_path) in self.dataset)
        pool = Pool(128)
        all_args = []
        for datafile, txt_path, ann_path in self.dataset:
            all_args.append([datafile, txt_path, ann_path])
        segments_file = pool.map(self.process_file, all_args)
        pool.close()
        pool.join()
        count = 0
        for i in range(len(segments_file)):
            count  = count + len(segments_file[i]['label'])
        print(count)
        # print(segments_file[5]['track'], segments_file[5]['label'])

        for segment in segments_file:
            # Add lists of segments to the segments object for the dataset
            self.segments['seg_preceding'].extend(segment['preceding'])
            self.segments['seg_concept1'].extend(segment['concept1'])
            self.segments['seg_middle'].extend(segment['middle'])
            self.segments['seg_concept2'].extend(segment['concept2'])
            self.segments['seg_succeeding'].extend(segment['succeeding'])
            self.segments['sentence'].extend(segment['sentence'])
            self.segments['track'].extend(segment['track'])
            # if not self.test:
            self.segments['label'].extend(segment['label'])


        # print("LENGTH of SEGMENT", len(self.segments))

            # To add lists of segments to the segments object for the dataset while maintaining the list separate
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_concept1'].append(segment['concept1'])
            # self.segments['seg_middle'].append(segment['middle'])
            # self.segments['seg_concept2'].append(segment['concept2'])
            # self.segments['seg_succeeding'].append(segment['succeeding'])
            # self.segments['sentence'].append(segment['sentence'])
            # self.segments['label'].append(segment['label'])

        if not self.test:
            # print(set(self.segments['label']))
            # print the number of instances of each relation classes
            print([(i, self.segments['label'].count(i)) for i in set(self.segments['label'])])

        # write the segments to a file
        file.list_to_file('sentence_train', self.segments['sentence'])
        file.list_to_file('preceding_seg', self.segments['seg_preceding'])
        file.list_to_file('concept1_seg', self.segments['seg_concept1'])
        file.list_to_file('middle_seg', self.segments['seg_middle'])
        file.list_to_file('concept2_seg', self.segments['seg_concept2'])
        file.list_to_file('succeeding_seg', self.segments['seg_succeeding'])
        file.list_to_file('track', self.segments['track'])
        # if not self.test:
        file.list_to_file('labels_train', self.segments['label'])

    def process_file(self,dataset):
    # def process_file(self,datafile, txt_path, ann_path):

        # for datafile, txt_path, ann_path in dataset:


        self.file = dataset[0]
        self.ann_path = dataset[2]
        self.txt_path = dataset[1]
        self.ann_obj = Annotation(self.ann_path)

        print("File", self.file)
        content = open(self.txt_path).read()
        # content_text = normalization.replace_Punctuation(content)

        self.doc = self.nlp_model(content)

        file_name = str(self.file )+".ann"
        if self.write_Predictions:
            if self.prediction_folder is not None:
                write_entities_to_file(self.ann_obj, file_name, self.prediction_folder)
            else:
                print("Define the path to the folder to save predictions ")

        segment = self.get_Segments_from_sentence(self.ann_obj)
        # print(self.file,segment['label'])
        # print(segment['track'])
        return segment

    def get_Segments_from_relations(self, ann):

        """
        For each relation object, it identifies the label and the entities first, then extracts the span of the
        entities from the text file using the start and end character span of the entities. Then it finds the
        sentence the entities are located in and passes the sentence and the spans of the entities to the function
        that extracts the following segments:

        Preceding - (tokenize words before the first concept)
        concept 1 - (tokenize words in the first concept)
        Middle - (tokenize words between 2 concepts)
        concept 2 - (tokenize words in the second concept)
        Succeeding - (tokenize words after the second concept)

        :param ann: annotation object
        :return: segments and label
        """

        # object to store the segments of a relation object
        segment = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                   'label': []}

        for label_rel, entity1, entity2 in ann.annotations['relations']:

            start_C1 = ann.annotations['entities'][entity1][1]
            end_C1 = ann.annotations['entities'][entity1][2]

            start_C2 = ann.annotations['entities'][entity2][1]
            end_C2 = ann.annotations['entities'][entity2][2]

            # to get arrange the entities in the order they are located in the sentence
            if start_C1 < start_C2:
                concept_1 = self.doc.char_span(start_C1, end_C1)
                concept_2 = self.doc.char_span(start_C2, end_C2)
            else:
                concept_1 = self.doc.char_span(start_C2, end_C2)
                concept_2 = self.doc.char_span(start_C1, end_C1)

            if concept_1 is not None and concept_2 is not None:
                # get the sentence where the entity is located
                sentence_C1 = str(concept_1.sent)
                sentence_C2 = str(concept_2.sent)
            else:
                break

            # if both entities are located in the same sentence return the sentence or
            # concatenate the individual sentences where the entities are located in to one sentence

            if sentence_C1 == sentence_C2:
                sentence = sentence_C1
            else:
                sentence = sentence_C1 + " " + sentence_C2

            sentence = normalization.remove_Punctuation(str(sentence).strip())
            concept_1 = normalization.remove_Punctuation(str(concept_1).strip())
            concept_2 = normalization.remove_Punctuation(str(concept_2).strip())
            segment['concept1'].append(concept_1)
            segment['concept2'].append(concept_2)
            segment['sentence'].append(sentence.replace('\n', ' '))

            preceding, middle, succeeding = extract_Segments(sentence, concept_1, concept_2)
            segment['preceding'].append(preceding.replace('\n', ' '))
            segment['middle'].append(middle.replace('\n', ' '))
            segment['succeeding'].append(succeeding.replace('\n', ' '))
            segment['label'].append(label_rel)

        return segment

    def get_Segments_from_sentence(self, ann):

        """

        In the annotation object, it identifies the sentence each problem entity is located and tries to determine
        the relations between other problem entities and other entity types in the same sentence. When a pair of
        entities is identified first it checks whether a annotated relation type exists, in that case it labels with
        the given annotated label if not it labels as a No - relation pair. finally it passes the sentence and the
        spans of the entities to the function that extracts the following segments:

        Preceding - (tokenize words before the first concept)
        concept 1 - (tokenize words in the first concept)
        Middle - (tokenize words between 2 concepts)
        concept 2 - (tokenize words in the second concept)
        Succeeding - (tokenize words after the second concept)

        :param ann: annotation object
        :return: segments and label: preceding, concept_1, middle, concept_2, succeeding, label

        """
        # object to store the segments of a relation object for a file
        doc_segments = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                        'label': [], 'track': []}

        # list to store the identified relation pair when both entities are same
        self.entity_holder = []

        for key1, value1 in ann.annotations['entities'].items():
            label1, start1, end1, mention1 = value1
            # when relations are dependent on one entity (dominant)
            if not self.generalize:
                # dominant label
                if label1 == self.rel_labels[0]:
                # if label1 == self.rel_labels[0] or label1 == self.rel_labels[1]:
                    for key2, value2 in ann.annotations['entities'].items():
                        label2, start2, end2, mention2 = value2
                        token = True
                        # if relation exists between the same entities
                        if self.same_entity_relation and label2 == self.rel_labels[0] and key1 != key2:
                            #needs checking
                            if self.test:
                                label_rel = "No Label"
                                segment = self.extract_sentences(ann, key2, key1, label_rel)
                                if segment is not None:
                                    doc_segments = add_file_segments(doc_segments, segment)
                            else:
                                for label_rel, entity1, entity2 in ann.annotations['relations']:
                                    if key2 == entity1 and key1 == entity2:
                                        segment = self.extract_sentences(ann, entity1, entity2, label_rel, True)
                                        doc_segments = add_file_segments(doc_segments, segment)
                                        token = False
                                        break

                                # No relations for the same entity
                                if token and self.no_rel_label:
                                    label_rel = self.no_rel_label[0]
                                    segment = self.extract_sentences(ann, key2, key1, label_rel)
                                    if segment is not None:
                                        doc_segments = add_file_segments(doc_segments, segment)
                        # when the entity pair do not contain entities of the same type
                        for i in range(len(self.rel_labels) - 1):
                            # match the dominant entity with other entities
                            if not self.same_entity_relation and label2 == self.rel_labels[i + 1]:#label2 - second entity label
                                if self.test:
                                    label_rel = "No Label"
                                    segment = self.extract_sentences(ann, key2, key1, label_rel)
                                    if segment is not None:
                                        doc_segments = add_file_segments(doc_segments, segment)
                                else:
                                    # for the relations that exist in the ann files
                                    for label_rel, entity1, entity2 in ann.annotations['relations']:
                                        # if key2 == entity2 and key1 == entity1:
                                        if key2 == entity1 and key1 == entity2:
                                        # when a match with an existing relation is found
                                            segment = self.extract_sentences(ann, entity1, entity2, label_rel, True)
                                            doc_segments = add_file_segments(doc_segments, segment)
                                            token = False
                                            break

                                    # No relations for the different entities
                                    if token and self.no_rel_label:
                                        label_rel = self.no_rel_label[0]
                                        segment = self.extract_sentences(ann, key2, key1, label_rel)
                                        if segment is not None:
                                            doc_segments = add_file_segments(doc_segments, segment)

            else:
                # when relation exists between all entity pairs
                for key2, value2 in ann.annotations['entities'].items():
                    label2, start2, end2, mention2 = value2
                    token = True

                    # for the same entity
                    if self.same_entity_relation and label2 == self.rel_labels[0] and key1 != key2:
                        if self.test:
                            label_rel = "No Label"
                            segment = self.extract_sentences(ann, key2, key1, label_rel)
                            if segment is not None:
                                doc_segments = add_file_segments(doc_segments, segment)
                        else:
                            #when relation exists in the ann file
                            for label_rel, entity1, entity2 in ann.annotations['relations']:
                                if key2 == entity1 and key1 == entity2:
                                    segment = self.extract_sentences(ann, entity1, entity2, label_rel, True)
                                    doc_segments = add_file_segments(doc_segments, segment)
                                    token = False
                                    break

                            # No relations for the same entity
                            if token and self.no_rel_label:
                                label_rel = self.no_rel_label[0]
                                segment = self.extract_sentences(ann, key2, key1, label_rel)
                                if segment is not None:
                                    doc_segments = add_file_segments(doc_segments, segment)

                    for i in range(len(self.rel_labels) - 1):
                        # for the different entities
                        if not self.same_entity_relation and label2 == self.rel_labels[i + 1]:
                            if self.test:
                                label_rel = "No Label"
                                segment = self.extract_sentences(ann, key2, key1, label_rel)
                                if segment is not None:
                                    doc_segments = add_file_segments(doc_segments, segment)
                            else:
                                # when relation exists in the ann file
                                for label_rel, entity1, entity2 in ann.annotations['relations']:
                                    if key2 == entity1 and key1 == entity2:
                                        segment = self.extract_sentences(ann, entity1, entity2, label_rel, True)
                                        doc_segments = add_file_segments(doc_segments, segment)
                                        token = False
                                        break

                                # No relations for the different entities
                                if token and self.no_rel_label:
                                    label_rel = self.no_rel_label[0]
                                    segment = self.extract_sentences(ann, key2, key1, label_rel)
                                    if segment is not None:
                                        doc_segments = add_file_segments(doc_segments, segment)

        return doc_segments

    def extract_sentences(self, ann, entity1, entity2, label_rel=None, join_sentences=False):
        """
        when the two entities are give as input, it identifies the sentences they are located and determines whether the
        entity pair is in the same sentence or not. if not they combine the sentences if there an annotated relation exist
        and returns None if an annotated relation doesn't exist
        :param ann: annotation object
        :param label_rel: relation type
        :param entity1: first entity in the considered pair
        :param entity2: second entity in the considered pair
        :param join_sentences: check for annotated relation in the data
        :return: segments and sentences and label
        """
        segment = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                   'label': [], 'track':[]}

        start_C1 = ann.annotations['entities'][entity1][1]
        end_C1 = ann.annotations['entities'][entity1][2]

        start_C2 = ann.annotations['entities'][entity2][1]
        end_C2 = ann.annotations['entities'][entity2][2]

        # to get arrange the entities in the order they are located in the sentence
        if start_C1 < start_C2:
            concept_1 = self.doc.char_span(start_C1, end_C1)
            concept_2 = self.doc.char_span(start_C2, end_C2)
        else:
            concept_1 = self.doc.char_span(start_C2, end_C2)
            concept_2 = self.doc.char_span(start_C1, end_C1)

        if concept_1 is not None and concept_2 is not None:
            # get the sentence the entities are located
            sentence_C1 = str(concept_1.sent.text)
            sentence_C2 = str(concept_2.sent.text)

            # if both entities are located in the same sentence return the sentence or concatenate the individual sentences where the entities are located in to one sentence
            if join_sentences:
                if sentence_C1 == sentence_C2:
                    sentence = sentence_C1
                else:
                    sentence = sentence_C1 + " " + sentence_C2
            else:
                #if the entity pair considered do not come from an annotated relation, strictly restrict to one sentence
                if sentence_C1 == sentence_C2:
                    sentence = sentence_C1
                    entity_pair = entity1 + '-' + entity2
                    # to make sure the same entity pair is not considered twice
                    if entity_pair not in self.entity_holder:
                        self.entity_holder.append(entity2 + '-' + entity1)
                    else:
                        sentence = None
                else:
                    sentence = None
        else:
            sentence = None

        if sentence is not None:
            sentence = normalization.remove_Punctuation(str(sentence).strip())
            concept_1 = normalization.remove_Punctuation(str(concept_1).strip())
            concept_2 = normalization.remove_Punctuation(str(concept_2).strip())
            preceding, middle, succeeding = extract_Segments(sentence, concept_1, concept_2)

            # remove the next line character in the extracted segment by replacing the '\n' with ' '
            segment['concept1'].append(concept_1.replace('\n', ' '))
            segment['concept2'].append(concept_2.replace('\n', ' '))
            segment['sentence'].append(sentence.replace('\n', ' '))
            segment['preceding'].append(preceding.replace('\n', ' '))
            segment['middle'].append(middle.replace('\n', ' '))
            segment['succeeding'].append(succeeding.replace('\n', ' '))
            segment['label'].append(label_rel)
            # print( int(self.file),int(entity1[1:]),int(entity2[1:]))
            segment['track'].append(int(self.file))
            segment['track'].append(int(entity1[1:]))
            segment['track'].append(int(entity2[1:]))
            # segment['track'].append(track)
        return segment
