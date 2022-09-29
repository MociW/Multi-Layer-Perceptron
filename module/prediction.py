import numpy as np
import os            # for opening file or directory
import pandas as pd  # for load and process raw dataset
import random        # for making random number
from pandas.plotting import scatter_matrix  # for scatter plotting visualization
import matplotlib.pyplot as plt


def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


classes = ['setosa    ', 'versicolor', 'virginica ']


def accuracy_average(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def f1_score(label, confusion_matrix):
    num = precision(label, confusion_matrix) * recall(label, confusion_matrix)
    denum = precision(label, confusion_matrix) + \
        recall(label, confusion_matrix)
    return 2 * (num/denum)


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def f1_score_average(confusion_matrix):
    num = precision_macro_average(
        confusion_matrix) * recall_macro_average(confusion_matrix)
    denum = precision_macro_average(
        confusion_matrix) + recall_macro_average(confusion_matrix)
    return 2 * (num/denum)


def label(confusion_matrix):
    print("label      precision  recall  f1_score")
    for index in range(len(classes)):
        print(f"{classes[index]} {precision(index, confusion_matrix):9.3f} {recall(index, confusion_matrix):6.3f}  {f1_score(index, confusion_matrix):6.3f}")
    