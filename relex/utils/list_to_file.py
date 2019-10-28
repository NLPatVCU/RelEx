# Author - Samantha Mahendran for RelEx


def list_to_file(file, input_list):
    """
    Method  to write the contents of a list to a file

    :param file: name of the output file.
    :param input_list: list needs to be written to file
    """

    with open(file, 'w') as f:
        for item in input_list:
            f.write("%s\n" % item)
