# Author - Samantha Mahendran for RelEx
import os


class Read_From_file:
    def __init__(self, file):
        """
        Reads external files and insert the content to a list. It also removes whitespace
        characters like `\n` at the end of each line

        :param file: name of the input file.
        """
        self.file = file

        if not os.path.isfile(self.file):
            raise FileNotFoundError("Not a valid file path")

        with open(self.file) as f:
            content = f.readlines()
        self.content = [x.strip() for x in content]
