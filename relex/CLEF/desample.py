from sklearn.utils import shuffle
from random import seed
from random import randint
import sys
import os


def desample(labels, sentences, ratio1, ratio2, ratio1_label, ratio2_label):
    """
    :param labels: array of labels
    :param sentences: array of sentences
    :param ratio1: goal ratio of the label declared in ratio1_label
    :param ratio2: goal ratio of the label declared in ratio2_label
    :param ratio1_label: label that corresponds to ratio1
    :param ratio2_label: label that corresponds to ratio2

    :return: desampled sentences, desampled labels

    """

    # shuffles labels
    labels_shuffled, sentences_shuffled = shuffle(labels, sentences)
    # gets current ratio for label2
    current_ratio2 = labels.count(ratio2_label)/(labels.count(ratio1_label))

    labels_shuffled_ratio2 = []
    sentences_shuffled_ratio2 = []
    labels_shuffled_other = []
    sentences_shuffled_other = []

    for x, y in zip(sentences_shuffled, labels_shuffled):
        if y == ratio2_label:
            sentences_shuffled_ratio2.append(x)
            labels_shuffled_ratio2.append(y)
        else:
            sentences_shuffled_other.append(x)
            labels_shuffled_other.append(y)

    # calculates number to remove
    num_to_remove = int(((current_ratio2-ratio2)/current_ratio2)*labels.count(ratio2_label))
    # removes correct number
    for x in range(0, num_to_remove):
        seed()
        index = randint(-1, len(labels_shuffled_ratio2)-1)
        labels_shuffled_ratio2.pop(index)
        sentences_shuffled_ratio2.pop(index)

    # recombines lists
    labels_desampled = labels_shuffled_ratio2 + labels_shuffled_other
    sentences_desampled = sentences_shuffled_ratio2 + sentences_shuffled_other

    # shuffles lists
    sentences_desampled_shuffled, labels_desampled_shuffled = shuffle(sentences_desampled, labels_desampled)

    return sentences_desampled_shuffled, labels_desampled_shuffled

def read_from_file(file):
    """
    Reads a file and returns its contents as a list
    :param file: path to file that will be read
    """

    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    return content

sentences = read_from_file("desample/concept1_seg")
labels = read_from_file("desample/labels_train")

def list_to_file(file, input_list):
    """
    Method  to write the contents of a list to a file.
    :param file: name of the output file.
    :param input_list: list needs to be written to file
    """

    with open(file, 'w') as f:
        for item in input_list:
            f.write("%s\n" % item)

sentences_shuffled, labels_shuffled = desample(labels, sentences, 1, 1, "STARTING_MATERIAL-REACTION_STEP", "No-Relation")

list_to_file("desampled/concept1_seg", sentences_shuffled)