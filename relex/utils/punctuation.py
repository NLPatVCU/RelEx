# Author - Samantha Mahendran for RelEx
class Punctuation:
    def __init__(self, string):
        """
        Class to remove or replace punctuation
        :param string: string that has the punctuation
        """
        self.string = string
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    def remove_punctuation(self):
        """
        method to remove punctuation from a given string. It traverses the given string
        and replaces the punctuation marks with null
        """
        for x in self.string.lower():
            if x in self.punctuations:
                self.string = self.string.replace(x, "")

    def replace_punctuation(self):
        """
        method to remove punctuation from a given string. It traverses the given string
        and replaces the punctuation marks with comma (,)
        """
        for x in self.string.lower():
            if x in self.punctuations:
                self.string = self.string.replace(x, ",")
