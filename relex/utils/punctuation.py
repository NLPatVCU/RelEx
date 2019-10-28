# Author - Samantha Mahendran for RelEx
def remove_punctuation(string):
    """
    method to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with null

    @param string: string to remove punctuation from
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string


def replace_punctuation(string):
    """
    method to remove punctuation from a given string. It traverses the given string
    and replaces the punctuation marks with comma (,)

    @param string: string to replace punctuation from
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, ",")
    return string
