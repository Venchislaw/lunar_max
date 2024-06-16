import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Confusion Matricies!
def confusion_matrix(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp += 1
        elif y_true[i] == y_pred[i] == 0:
            tn += 1
        elif (y_true[i] == 0) and (y_pred[i] == 1):
            fp += 1
        else:
            fn += 1

    return np.array([[tp, fp], [fn, tn]])


def plot_confusion_matrix(matrix):
    plt.title('Confusion Matrix')
    matrix = pd.DataFrame(matrix, index=[i for i in "10"], columns=[i for i in "10"])
    sns.heatmap(matrix, annot=True)
    plt.show()


def precision_recall(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def f_1_score(confusion_matrix):
    prec, rec = precision_recall(confusion_matrix)

    return 2 * prec * rec / (prec + rec)

matrix = confusion_matrix([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
plot_confusion_matrix(matrix)
