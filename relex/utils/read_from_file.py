# Author - Samantha Mahendran for RelEx
import os

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
