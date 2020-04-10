# Author - Samantha Mahendran for remove_punctuation, replace_punctuation
# Author - Cora Lewis for desample, gcd, desample_given_unique_labels_current_ratio, find_unique_labels_props
import random
from functools import reduce


def remove_Punctuation(string):
    """
    method to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with null
    :param string: string to remove punctuation from
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, '')
    return string


def replace_Punctuation(string):
    """
    method to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with comma (,)
    :param string: string to replace punctuation from
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, ',')
    return string


def desample_given_unique_labels_current_ratios(sentences, labels, ratios, unique_labels, current_ratios):
    """
    Function desamples sentences and labels based on goal ratios, unique labels, and current ratios
    :param sentences: list of sentences
    :param labels: list of labels
    :param ratios: list of goal ratios in order of occurrence in param labels
    :param unique_labels: list of unique labels in order of occurrence in param labels
    :param current_ratios: list of current ratios in order of occurrence in param labels
    :return: tuple that contains desampled lables and sentences
    """
    # finds amounts of each label that will be in the new list
    new_amount = []
    for ratio, current, label in zip(ratios, current_ratios, unique_labels):
        reduce_num = ((current - ratio) / current) * labels.count(label)
        new_amount.append(labels.count(label) - reduce_num)

    # shuffles lists
    combined_list = list(zip(sentences, labels))
    random.shuffle(combined_list)
    shuffled_sentences, shuffled_labels = zip(*combined_list)
    shuffled_labels = list(shuffled_labels)
    shuffled_sentences = list(shuffled_sentences)

    # removes correct number of labels and sentences
    for num, num_label in zip(new_amount, unique_labels):
        while shuffled_labels.count(num_label) > num:
            shuffled_sentences.pop(shuffled_labels.index(num_label))
            shuffled_labels.remove(num_label)

    return shuffled_sentences, shuffled_labels


def gcd(n, d):
    """
    function calculates greatest common divisor
    :param n: numerator
    :param d: denominator
    :return: greatest common divisor
    """
    # returns 1 if there is no GCD
    if n % d != 0:
        return 1
    else:  # used euclidean algorithm to find GCD
        while d:
            n, d = d, n % d
            return n


def find_unique_labels_props(labels):
    """
    function finds labels and unique_labels from a list of labels
    :param labels: list containing  the labels for the data
    :return: array containing a list of the unique labels and current ratios of thoes labels in param labels
    """
    # finds unique labels
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    # finds current ratios
    props = [labels.count(label) for label in unique_labels]
    props_simple = [prop / reduce(gcd, props) for prop in props]
    return [unique_labels, props_simple]


def desample(sentences, labels, ratios):
    """
    function desamples 2 lists based on the ratios given
    :param sentences: list of the sentences
    :param labels: list of the labels
    :param ratios: list with each element being the ratio. Ratios should be in order of occurrence in param labels.
    :return: tuple containing desampled labels and sentences
    """

    # finds unique labels and current ratio of the labels
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    props = [labels.count(label) for label in unique_labels]
    current_ratios = [prop / reduce(gcd, props) for prop in props]

    # finds amounts of each label that will be in the new list
    new_amount = []
    for ratio, current, label in zip(ratios, current_ratios, unique_labels):
        if len(ratios) < 3 and ratio - current < 0:  # when there are only two labels,sometimes you have to get to the
            # reduce_num in a different manner
            reduce_num = (((current - ratio) / current) * labels.count(label)) + (current - ratio)
        else:
            reduce_num = ((current - ratio) / current) * labels.count(label)
        new_amount.append(labels.count(label) - reduce_num)

    # shuffles lists
    combined_list = list(zip(sentences, labels))
    random.shuffle(combined_list)
    shuffled_sentences, shuffled_labels = zip(*combined_list)
    shuffled_labels = list(shuffled_labels)
    shuffled_sentences = list(shuffled_sentences)

    # removes correct number of labels and sentences
    for num, num_label in zip(new_amount, unique_labels):
        while shuffled_labels.count(num_label) > num:
            shuffled_sentences.pop(shuffled_labels.index(num_label))
            shuffled_labels.remove(num_label)
            print(shuffled_labels)

    return shuffled_sentences, shuffled_labels