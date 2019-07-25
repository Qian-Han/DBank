import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing

final_results_mat = pd.DataFrame()
final_results_mat['dataset'] = []
final_results_mat['class_size'] = []
final_results_mat['classifier'] = []
final_results_mat['kfold_type'] = []
final_results_mat['F1'] = []
final_results_mat['Precision'] = []
final_results_mat['Recall'] = []
final_results_mat['AUC'] = []
final_results_mat['FPR'] = []
final_results_mat['FNR'] = []

def GetResultRow(y_true_total, y_pred_total, y_prob_total, dataset_name, algorithm_name, l_to_consider, kfold_name, class_size):
    
    AUC = 0

    try:
        AUC = sklearn.metrics.roc_auc_score(y_true_total,y_prob_total)
    except Exception as e:
        raise Exception('Error while computing AUC:', e)
   
    result = {}
    result['dataset'] = dataset_name
    result['class_size'] = class_size
    result['classifier'] = algorithm_name
    result['kfold_type'] = kfold_name
    result['F1'] = sklearn.metrics.f1_score(y_true_total,y_pred_total, average='binary')
    result['Precision'] = sklearn.metrics.precision_score(y_true_total,y_pred_total, average='binary')
    result['Recall'] = sklearn.metrics.recall_score(y_true_total, y_pred_total, average='binary')
    result['AUC'] = AUC
   
    confusion_matrix_total = sklearn.metrics.confusion_matrix(np.array(y_true_total), np.array(y_pred_total))
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_total, y_pred_total).ravel()
    result['FPR'] = fp / float(fp + tn)
    result['FNR'] = fn / float(tp + fn)
    print('\tPrecision: {} - Recall: {} - F1: {} - AUC: {}'.format(result['Precision'], result['Recall'], result['F1'], result['AUC']))
    cm_total = pd.DataFrame(confusion_matrix_total)
    cm_sum = cm_total.sum(axis = 1).astype('float')
    cm_total_normalized = cm_total.divide(cm_sum, axis = 0)

    return cm_total_normalized, result
