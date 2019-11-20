# Author : Samantha Mahendran for RelEx

""" This file splits the dataset into sentences to be used in Segment CNN. """
from data import Annotation
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
import spacy
from utils import list_to_file, remove_punctuation, replace_punctuation

def add_file_segments(doc_segments, segment):
    """
    Function to add the local segment object to the global segment object

    :param doc_segments: global segment object
    :param segment: local segment object
    """
    doc_segments['preceding'].extend(segment['preceding'])
    doc_segments['concept1'].extend(segment['concept1'])
    doc_segments['middle'].extend(segment['middle'])
    doc_segments['concept2'].extend(segment['concept2'])
    doc_segments['succeeding'].extend(segment['succeeding'])
    doc_segments['sentence'].extend(segment['sentence'])
    doc_segments['label'].extend(segment['label'])

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
    """

    preceding = sentence[0:sentence.find(span1)]
    preceding = remove_punctuation(str(preceding)).strip()

    middle = sentence[sentence.find(span1) + len(span1):sentence.find(span2)]
    middle = remove_punctuation(str(middle)).strip()

    succeeding = sentence[sentence.find(span2) + len(span2):]
    succeeding = remove_punctuation(str(succeeding)).strip()

    return preceding, middle, succeeding


class Segmentation:
    def __init__(self, dataset=None, rel_labels=None, no_rel_label=None, sentence_align=False, test=False,
                 same_entity_relation=False, de_sample=None):
        """
        This class segments the dataset so the data can be used in Segment CNN.

        :param datatset: path to datatset
        :param rel_labels: relationship labels array
        :param no_rel_labels: array with label for when there is no relation
        :param sentence_align: sentence align flag
        :param test: test flag
        :param same_entity_relation: sentence entity relation flag
        :param de_sample: de_sample
        """

        self.dataset = dataset
        # list of the entities of relations
        self.rel_labels = rel_labels
        self.test = test
        # if relation exists between same entity
        self.same_entity_relation = same_entity_relation
        self.nlp_model = English()

        # if there is no relation between entities in a sentence
        if no_rel_label:
            self.no_rel_label = no_rel_label
        else:
            self.no_rel_label = False

        # desampling - Reduce the size of the dataset to solve the data imbalance problem
        if de_sample:
            self.de_sample = de_sample
        else:
            self.de_sample = False

        """
        Simple pipeline component, to allow custom sentence boundary detection logic that doesn’t require the dependency parse.
        A simpler, rule-based strategy that doesn’t require a statistical model to be loaded
        """
        if sentence_align:
            sentencizer = Sentencizer(punct_chars=["\n"])
        else:
            sentencizer = Sentencizer(punct_chars=["\n", ".", "?"])

        self.nlp_model.add_pipe(sentencizer)

        # self.nlp_model = spacy.load('en_core_web_sm')

        # global segmentation object that returns all segments and the label
        self.segments = {'seg_preceding': [], 'seg_concept1': [], 'seg_concept2': [], 'seg_middle': [],
                         'seg_succeeding': [], 'sentence': [], 'label': []}

        for datafile, txt_path, ann_path in dataset:
            print(datafile)

            self.ann_path = ann_path
            self.txt_path = txt_path
            self.ann_obj = Annotation(self.ann_path)

            content = open(self.txt_path).read()
            # content_text = replace_Punctuation(content)

            self.doc = self.nlp_model(content)

            # returns the segments from the sentences
            segment = self.get_Segments_from_sentence(self.ann_obj)
            # segment = self.get_Segments_from_relations(self.ann_obj )

            # Add lists of segments to the segments object for the dataset
            self.segments['seg_preceding'].extend(segment['preceding'])
            self.segments['seg_concept1'].extend(segment['concept1'])
            self.segments['seg_middle'].extend(segment['middle'])
            self.segments['seg_concept2'].extend(segment['concept2'])
            self.segments['seg_succeeding'].extend(segment['succeeding'])
            self.segments['sentence'].extend(segment['sentence'])
            if not self.test:
                self.segments['label'].extend(segment['label'])

            # To add lists of segments to the segments object for the dataset while maintaining the list separate
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_preceding'].append(segment['preceding'])
            # self.segments['seg_concept1'].append(segment['concept1'])
            # self.segments['seg_middle'].append(segment['middle'])
            # self.segments['seg_concept2'].append(segment['concept2'])
            # self.segments['seg_succeeding'].append(segment['succeeding'])
            # self.segments['sentence'].append(segment['sentence'])
            # self.segments['label'].append(segment['label'])

        # if not self.test:
        #     print(set(self.segments['label']))
        #     # print the number of instances of each relation classes
        #     print([(i, self.segments['label'].count(i)) for i in set(self.segments['label'])])

        # write the segments to a file
        list_to_file('sentence_train', self.segments['sentence'])
        list_to_file('preceding_seg', self.segments['seg_preceding'])
        list_to_file('concept1_seg', self.segments['seg_concept1'])
        list_to_file('middle_seg', self.segments['seg_middle'])
        list_to_file('concept2_seg', self.segments['seg_concept2'])
        list_to_file('succeeding_seg', self.segments['seg_succeeding'])
        if not self.test:
            list_to_file('labels_train', self.segments['label'])

    #currently not used
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

            sentence = remove_punctuation(str(sentence).strip())
            concept_1 = remove_punctuation(str(concept_1).strip())
            concept_2 = remove_punctuation(str(concept_2).strip())
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
        In the annotation object, it identifies the sentence each problem entity (i2b2) is located and tries to determine
        the relations between other problem entities and other entity types in the same sentence. When a pair of
        entities is identified first it checks whether an annotated relation type already exists,
            yes : label it with the given annotated label
            no : if no-rel label is active label as a No - relation pair
        Finally it passes the sentence and the spans of the entities to the function that extracts the following segments:
        Preceding - (tokenize words before the first concept)
        concept 1 - (tokenize words in the first concept)
        Middle - (tokenize words between 2 concepts)
        concept 2 - (tokenize words in the second concept)
        Succeeding - (tokenize words after the second concept)

        :param ann: annotation object
        """
        # object to store the segments of a relation object for a file
        doc_segments = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                        'label': []}

        # list to store the identified relation pair when both entities are same
        self.entity_holder = []

        for key1, value1 in ann.annotations['entities'].items():
            label1, start1, end1, mention1 = value1

            # for problem entitiyy (i2b2)
            if label1 == self.rel_labels[0]:
                for key2, value2 in ann.annotations['entities'].items():
                    label2, start2, end2, mention2 = value2
                    token = True

                    # for the same entity
                    if self.same_entity_relation and label2 == self.rel_labels[0] and key1 != key2:
                        if self.test:
                            label_rel = None
                            segment = self.extract_sentences(ann, key1, key2, label_rel)
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
                                segment = self.extract_sentences(ann, key1, key2, label_rel)
                                if segment is not None:
                                    doc_segments = add_file_segments(doc_segments, segment)

                    for i in range(len(self.rel_labels) - 1):
                        # for the different entities
                        if not self.same_entity_relation and label2 == self.rel_labels[i + 1]:
                            if self.test:
                                label_rel = None
                                segment = self.extract_sentences(ann, key1, key2, label_rel)
                                if segment is not None:
                                    doc_segments = add_file_segments(doc_segments, segment)
                            else:
                                for label_rel, entity1, entity2 in ann.annotations['relations']:
                                    if key2 == entity1 and key1 == entity2:
                                        segment = self.extract_sentences(ann, entity1, entity2, label_rel, True)
                                        doc_segments = add_file_segments(doc_segments, segment)
                                        token = False
                                        break

                                # No relations for the different entities
                                if token and self.no_rel_label:
                                    label_rel = self.no_rel_label[i]
                                    segment = self.extract_sentences(ann, key1, key2, label_rel)
                                    if segment is not None:
                                        doc_segments = add_file_segments(doc_segments, segment)

        return doc_segments

    def extract_sentences(self, ann, entity1, entity2, label_rel=None, from_relation=False):
        """
        when the two entities are give as input, it identifies the sentences they are located and determines whether the
        entity pair is in the same sentence or not. if not they combine the sentences if there an annotated relation exist
        and returns None if an annotated relation doesn't exist

        :param ann: annotation object
        :param label_rel: relation type
        :param entity1: first entity in the considered pair
        :param entity2: second entity in the considered pair
        :param from_relation: check for annotated relation in the data
        """
        segment = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],
                   'label': []}
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

            # get the sentence the entity is located
            sentence_C1 = str(concept_1.sent.text)
            sentence_C2 = str(concept_2.sent.text)

            # if both entities are located in the same sentence return the sentence or
            # concatenate the individual sentences where the entities are located in to one sentence
            if from_relation:
                if sentence_C1 == sentence_C2:
                    sentence = sentence_C1
                else:
                    sentence = sentence_C1 + " " + sentence_C2
            else:
                if sentence_C1 == sentence_C2:
                    sentence = sentence_C1
                    entity_pair = entity1 + '-' + entity2
                    if entity_pair not in self.entity_holder:
                        self.entity_holder.append(entity2 + '-' + entity1)
                    else:
                        sentence = None
                else:
                    sentence = None
        else:
            sentence = None

        if sentence is not None:
            sentence = remove_punctuation(str(sentence).strip())
            concept_1 = remove_punctuation(str(concept_1).strip())
            concept_2 = remove_punctuation(str(concept_2).strip())
            preceding, middle, succeeding = extract_Segments(sentence, concept_1, concept_2)

            # remove the next line character in the extracted segment by replacing the '\n' with ' '
            segment['concept1'].append(concept_1.replace('\n', ' '))
            segment['concept2'].append(concept_2.replace('\n', ' '))
            segment['sentence'].append(sentence.replace('\n', ' '))
            segment['preceding'].append(preceding.replace('\n', ' '))
            segment['middle'].append(middle.replace('\n', ' '))
            segment['succeeding'].append(succeeding.replace('\n', ' '))
            segment['label'].append(label_rel)

        return segment
