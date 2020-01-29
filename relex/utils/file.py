# Author - Samantha Mahendran for list_to_file, read_from_file
# Author - Cora Lewis for output_to_file
import os
from sklearn.metrics import classification_report
import pandas as pd


def list_to_file(file, input_list):
    """
    Method  to write the contents of a list to a file.
    :param file: name of the output file.
    :param input_list: list needs to be written to file
    """

    with open(file, 'w') as f:
        for item in input_list:
            f.write("%s\n" % item)


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


def output_to_file( true_values, pred_values, output_path, target):
    """
    Method  to create .txt file and csv file of classification report
    :param true_values: correct labels for dataset
    :param pred_values: labels predicted by model
    :param output_path: path to where txt file of results should be created
    :param target: label types
    """
    report = classification_report(true_values, pred_values, target_names=target)
    report_dict = classification_report(true_values, pred_values, target_names=target, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # writes .txt file with results
    txt_file = open(output_path, 'a')
    txt_file.write(report)
    txt_file.close()

    # writes csv file
    csv_report = df_report.to_csv()
    csv_file = open(output_path, 'a')
    csv_file.write(csv_report)
    csv_file.close()
