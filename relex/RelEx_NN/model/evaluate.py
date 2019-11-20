# Author : Samantha Mahendran for RelEx

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate
from statistics import mean
import numpy as np


def predict(model, x_test, y_test, encoder_classes):
    """
    Takes the predictions as input and returns the indices of the maximum values along an axis using numpy argmax function as true labels.
    Then evaluates it against the trained model
    :param model: trained model
    :param x_test: test data
    :param y_test: test true labels
    :param encoder_classes:
    :return: predicted and true labels
    """
    pred = model.predict(x_test)
    y_pred_ind = np.argmax(pred, axis=1)
    y_true_ind = np.argmax(y_test, axis=1)
    y_pred = [encoder_classes[i] for i in y_pred_ind]
    y_true = [encoder_classes[i] for i in y_true_ind]
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Accuracy :", test_acc)
    print("Loss : ", test_loss)

    return y_pred, y_true


def evaluate_Model(y_pred, y_true, encoder_classes):
    """
    Prints a classification report and the f1 scores 
    :param y_pred:
    :param y_true:
    :param encoder_classes:
    """
    print(classification_report(y_true, y_pred, target_names=encoder_classes))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='weighted'))

    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)


def cv_evaluation_fold(y_pred, y_true, labels):
    """
    Evaluation metrics for each fold
    :param y_pred: predicted labels
    :param y_true: true labels
    :param labels: list of the classes
    :return:
    """
    fold_statistics = {}
    for label in labels:
        fold_statistics[label] = {}
        f1 = f1_score(y_true, y_pred, average='micro', labels=[label])
        precision = precision_score(y_true, y_pred, average='micro', labels=[label])
        recall = recall_score(y_true, y_pred, average='micro', labels=[label])
        fold_statistics[label]['precision'] = precision
        fold_statistics[label]['recall'] = recall
        fold_statistics[label]['f1'] = f1

    # add averages
    fold_statistics['system'] = {}
    f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    fold_statistics['system']['precision'] = precision
    fold_statistics['system']['recall'] = recall
    fold_statistics['system']['f1'] = f1

    table_data = [[label,
                   format(fold_statistics[label]['precision'], ".3f"),
                   format(fold_statistics[label]['recall'], ".3f"),
                   format(fold_statistics[label]['f1'], ".3f")]
                  for label in labels + ['system']]

    print(tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1'],
                   tablefmt='orgtbl'))
    return fold_statistics


def cv_evaluation(labels, evaluation_statistics):
    """
    considers the metrics of each fold and takes the average.
    :param labels: list of the classes
    :param evaluation_statistics: statistics
    """
    statistics_all_folds = {}

    for label in labels + ['system']:
        statistics_all_folds[label] = {}
        statistics_all_folds[label]['precision_average'] = mean(
            [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
        statistics_all_folds[label]['precision_max'] = max(
            [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])
        statistics_all_folds[label]['precision_min'] = min(
            [evaluation_statistics[fold][label]['precision'] for fold in evaluation_statistics])

        statistics_all_folds[label]['recall_average'] = mean(
            [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
        statistics_all_folds[label]['recall_max'] = max(
            [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])
        statistics_all_folds[label]['recall_min'] = min(
            [evaluation_statistics[fold][label]['recall'] for fold in evaluation_statistics])

        statistics_all_folds[label]['f1_average'] = mean(
            [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
        statistics_all_folds[label]['f1_max'] = max(
            [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])
        statistics_all_folds[label]['f1_min'] = min(
            [evaluation_statistics[fold][label]['f1'] for fold in evaluation_statistics])

    table_data = [[label,
                   format(statistics_all_folds[label]['precision_average'], ".3f"),
                   format(statistics_all_folds[label]['recall_average'], ".3f"),
                   format(statistics_all_folds[label]['f1_average'], ".3f"),
                   format(statistics_all_folds[label]['f1_min'], ".3f"),
                   format(statistics_all_folds[label]['f1_max'], ".3f")]
                  for label in labels + ['system']]

    print("\n" + tabulate(table_data, headers=['Relation', 'Precision', 'Recall', 'F1', 'F1_Min', 'F1_Max'],
                          tablefmt='orgtbl'))
