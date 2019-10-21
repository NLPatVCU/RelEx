# Author - Samantha Mahendran for RelEx
class List_To_File:
    def __init__(self, file, input_list):
        """
        Class  to write the contents of a list to a file

        :param file: name of the output file.
        :param input_list: list needs to be written to file
        """
        file = file
        input_list = input_list
        with open(file, 'w') as f:
            for item in input_list:
                f.write("%s\n" % item)
