# Author - Cora Lewis for RelEx
from sklearn.metrics import classification_report
import pandas as pd


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
