import sklearn.preprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.neighbors
from sklearn.pipeline import Pipeline
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.decomposition
import pandas as pd
import sys
import sklearn.cross_validation as cv


def getLabels(labels, family_to_consider=None):
    """
    This function returns a new binary classifier
    given a multiclass, in which the family to
    consider is mapped to one and the rest is mapped to zero
    """

    # Consider multi-class
    if family_to_consider == None:
        #         print('Returning labels for multi-class')
        return labels
    else:
        #         print('Returning labels=1 only for class #' + str(family_to_consider))
        new_labels = []
        for l in labels:
            if l == family_to_consider:
                new_labels.append(1)
            else:
                new_labels.append(0)
        return pd.Series(new_labels)


# # Run classifiers

# In[5]:

def compute_macro_stats(y_pred, y_true):
    precision_array = []
    recall_array = []

    class_set = set(y_true)
    L = len(y_true)

    # iterating through classes
    for current_class in class_set:
        mystats = {}
        mystats['fn'] = 0
        mystats['fp'] = 0
        mystats['tp'] = 0

        # for all elements of the prediction
        for i in range(L):

            # true positive
            if y_pred[i] == y_true[i] and (y_pred[i] == current_class):
                mystats['tp'] += 1

            # false negative
            if (y_true[i] == current_class) and (y_pred[i] != current_class):
                mystats['fn'] += 1

            # false positive
            if (y_true[i] != current_class) and (y_pred[i] == current_class):
                mystats['fp'] += 1


        precision = compute_precision(mystats)
        recall = compute_recall(mystats)

        if precision == -1000 or recall == -1000:
            continue

        # appending precision and recall
        precision_array.append(precision)
        recall_array.append(recall)

    precision_avg = float(np.sum(precision_array)) / (float(len(class_set)))
    recall_avg = float(np.sum(recall_array)) / (float(len(class_set)))

    return (precision_avg, recall_avg)


def compute_micro_stats(y_pred, y_true):
    mystats = {}
    mystats['fn'] = 0
    mystats['fp'] = 0
    mystats['tp'] = 0
    mystats['tn'] = 0
    class_set = set(y_true)
    L = len(y_pred)
    # iterating through classes
    for current_class in class_set:
        for i in range(L):

            # true positive
            if y_pred[i] == y_true[i] and (y_pred[i] == current_class):
                mystats['tp'] += 1

            # false negative
            if (y_true[i] == current_class) and (y_pred[i] != current_class):
                mystats['fn'] += 1

            # false positive
            if (y_true[i] != current_class) and (y_pred[i] == current_class):
                mystats['fp'] += 1

    # Precision and recall
    precision_avg = compute_precision(mystats)
    recall_avg = compute_recall(mystats)

    return (precision_avg, recall_avg)


def compute_precision(mystats):
    tp = mystats['tp']
    tpfp = mystats['tp'] + mystats['fp']

    if tpfp == 0:
        return -1000

    return (float(tp) / float(tpfp))


def compute_recall(mystats):
    tp = mystats['tp']
    tpfn = mystats['tp'] + mystats['fn']

    if tpfn == 0:
        return -1000

    return (float(tp) / (float(tpfn)))


def compute_fscore(precision, recall):
    if precision == 0 and recall == 0:
        print("WARNING: Precision and Recall are 0! Return Fscore=-1")
        return -1
    return 2.0 * ((float(precision * recall)) / (float(precision + recall)))


