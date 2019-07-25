import argparse
import time
import sklearn
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.decomposition
import sklearn.preprocessing
from sklearn import tree
from subprocess import call
import sys
import json
import numpy as np
import pandas as pd
import sklearn.feature_selection
from sklearn.model_selection import KFold
from collections import Counter
from libs.possible_classifiers import *
from libs.utils import *
import config
from libs.GetAllSR_fun import *
from libs.SUS_SR_feature_fun import GetAPIList
from libs.GetFinalResultRow_fun import *


DEFAULT_RESULTS_FOLDER = config.DEFAULT_RESULTS_FOLDER

DEFAULT_BASE_PATH = config.DEFAULT_BASE_PATH

pd.options.mode.chained_assignment = None


aparser = argparse.ArgumentParser(description='Runs various classifiers on Android Malware Features')

aparser.add_argument('--niterations', default=1, type=int, metavar='N',
                     help='The number of iterations to do')

aparser.add_argument('--nfolds', default=10, type=int, metavar='F',
                     help='The number of folds of the StratifiedKFold')

aparser.add_argument('--minsamples', default=10, type=int, metavar='MS',
                     help='Minimum number of samples to keep a family')

aparser.add_argument('--results', default=DEFAULT_RESULTS_FOLDER, type=str, metavar='RPATH',
                     help='The filepath of the results folder.')

aparser.add_argument('--datasets', default=['S','D','SD'], nargs='*', metavar='D',
                     help='The names of the datasets to consider.')
aparser.add_argument('--sr_method', default=['0'], nargs='*', metavar='MSR',
                     help='The method flag of sr and sus feature')
aparser.add_argument('--classifiers', default=['RF' 'DT' 'NB' 'LR' 'KNN' 'SVM', 'ADA', 'GBDT', 'NN_RBM','NN_MLP','XGB','KerasNN_MLP'], nargs='*', metavar='CL',
                     help='The names of the classifiers that are supported to run.')

aparser.add_argument('--kfolds', default=['Simple','Stratified'], nargs='*', metavar='KF',
                  help='The names of the K-Folds algorithm to run (NOTE THAT IF FAMILIES WITH LESS THAN 10 SAMPLES ARE IN CODE, ERROR SHOULD BE RAISED).')

aparser.add_argument('--features', default=DEFAULT_BASE_PATH, type=str, metavar='FPATH',
                    help='The basepath of the folder containing the features to use in this run.')

# Binary variables to set True or False some of them
aparser.add_argument('--save-cm', dest='cm', action='store_true', help='Save confusion matrix for all the results.')
aparser.add_argument('--save-proba', dest='proba', action='store_true', help='Save probability matrix of RF classifier.')
aparser.add_argument('--save-fi', dest='fi', action='store_true', help='Save feature importances.')
aparser.add_argument('--use-binary', dest='binary', action='store_true', help='Uses binary classification for the classifiers.')
aparser.add_argument('--use-pca', dest='pca', action='store_true', help='Uses PCA for feature reduction (keeps top-100 features).')
aparser.add_argument('--use-polynomial', dest='poly', action='store_true', help='Uses polynomial features (after keeping only the top-100 feature using feature selection).')
aparser.add_argument('--debug', dest='debug', action='store_true', help='Show debug information.')
aparser.set_defaults(debug=False,binary=False,fi=False,cm=False)
command_line_args = vars(aparser.parse_args())

# getting the arguments
NFOLDS = command_line_args['nfolds']
NITERATIONS = command_line_args['niterations']
RESULTS_FOLDER = command_line_args['results']
BASE_PATH = command_line_args['features']
DATASETS_TO_CONSIDER = command_line_args['datasets']
SHOW_DEBUG = command_line_args['debug']
SAVE_CONFUSION_MATRIX = command_line_args['cm']
SAVE_FEATURE_IMPORTANCES = command_line_args['fi']
SAVE_PROBABILITY_MATRIX = command_line_args['proba']
USE_BINARY_CLASSIFICATION = command_line_args['binary']
USE_PCA = command_line_args['pca']
USE_POLYNOMIAL = command_line_args['poly']
KFOLDS_TO_RUN = command_line_args['kfolds']
MIN_SAMPLES = command_line_args['minsamples']

SR_METHOD = command_line_args['sr_method']
import os
if not os.path.exists(RESULTS_FOLDER):
    print('*'*50)
    print("Creating results folder: {}".format(RESULTS_FOLDER))
    os.makedirs(RESULTS_FOLDER)

for fold in KFOLDS_TO_RUN:
    if fold not in ['Stratified','Simple']:
        raise Exception("This type of K-fold is not supported: {}".format(fold))

if USE_PCA and USE_POLYNOMIAL:
    raise Exception("Cannot use both PCA and Polynomial feature reduction")

if SHOW_DEBUG:
    print('Params:', command_line_args)

USE_GT2 = False

# Supported classifiers
CLASSIFIERS_DICTIONARY = {
    'SVM': run_svm,
    'KNN': run_k_nearest_neighbor,
    'LR': run_logistic_regression,
    'DT': run_decision_tree_clf,
    'NB': run_naive_bayes,
    'RF': run_random_forest,
    'ADA': run_adaboost,
    'GBDT': run_gradient_boosting,
    'NN_RBM': run_deeplearning,
    'NN_MLP': run_MLPClassifier,
    'KerasNN_MLP': run_KerasNNClassifier,
    'LINSVC': run_lin_svc
}


CLASSIFIERS_TO_RUN = {}
for cl in command_line_args['classifiers']:
    if cl not in CLASSIFIERS_DICTIONARY:
        print('ERROR: Classifier {} not supported!'.format(cl))
        sys.exit(0)
    CLASSIFIERS_TO_RUN[cl] = CLASSIFIERS_DICTIONARY[cl]

# prediction matrix
prediction_matrix = dict()
prediction_matrix['idx'] = np.array([])
prediction_matrix['y_true'] = np.array([])
prediction_matrix['y_pred'] = np.array([])
prediction_matrix['classifier'] = np.array([])
prediction_matrix['cross_val_iteration_number'] = np.array([])
prediction_matrix['dataset'] = np.array([])
prediction_matrix['n_iteration'] = np.array([])
prediction_matrix['considered_class'] = np.array([])
prediction_matrix['y_pred_proba_max'] = np.array([])
prediction_matrix['y_pred_proba_idxmax'] = np.array([])

# Constructing a results matrix
results_matrix = pd.DataFrame()

results_matrix['dataset'] = []
results_matrix['n_iteration'] = []
results_matrix['considered_class'] = []
results_matrix['class_size'] = []
results_matrix['classifier'] = []
results_matrix['kfold_type'] = []
results_matrix['cross_val_iteration_number'] = []
results_matrix['criteria_for_computing_scores'] = []

# can be both micro and macro

# Sklearn
results_matrix['MiF'] = []
results_matrix['MaF'] = []

results_matrix['MiAUC'] = []
results_matrix['MaAUC'] = []

results_matrix['MiPrecision'] = []
results_matrix['MaPrecision'] = []
results_matrix['MiRecall'] = []
results_matrix['MaRecall'] = []

results_matrix['MiF_custom'] = []
results_matrix['MaF_custom'] = []
results_matrix['MiPrecision_custom'] = []
results_matrix['MaPrecision_custom'] = []
results_matrix['MiRecall_custom'] = []
results_matrix['MaRecall_custom'] = []

feature_importance_matrix = pd.DataFrame()

feature_importance_matrix['dataset'] = []
feature_importance_matrix['n_iteration'] = []
feature_importance_matrix['considered_class'] = []
feature_importance_matrix['class_size'] = []
feature_importance_matrix['classifier'] = []
feature_importance_matrix['kfold_type'] = []
feature_importance_matrix['cross_val_iteration_number'] = []


for CURRENT_DATASET in DATASETS_TO_CONSIDER:


    if SHOW_DEBUG:
        print('#'*30)
        print('DATASET: {}'.format(CURRENT_DATASET))
        print('#'*30)

    FEATURES_PATH = None
    FEATURES_PATH = BASE_PATH + '{}.csv'.format(CURRENT_DATASET)

    print(FEATURES_PATH)
    print("*"*50)

    # loading samples
    print("Loading samples...")
    # Note: check that column is actually missing
    samples = pd.read_csv(FEATURES_PATH, index_col=False)

    # needed to encode author
    le = sklearn.preprocessing.LabelEncoder()
    if 'author' in samples.columns:
        print("Dropping author")
        # samples['author_INT'] = le.fit_transform(samples['author'])
        samples.drop('author',axis=1,inplace=True)
    print("Loaded.")

    if False:
        samples.to_csv('/tmp/samples_ordered.csv',index=False)
        # shuffling the samples, and resetting the index
        samples = samples.iloc[np.random.permutation(len(samples))]
        samples = samples.reset_index(drop=True)
        samples.to_csv('/tmp/samples_shuffle.csv',index=False)

    ###########################
    # preprocess for sr and sus feature generation, '0' means ignore sr and sus features
    # adjacent matrix of APK call graph
    aja_matrix = np.zeros([1,1])
    # level 17 std apks
    std_apis = []

    # API feature for current samples, used for generating SR and SUS features
    api_feature_with_label = pd.DataFrame()
    pd.options.display.max_rows = 200
    # 1-10,11-20,21-30....
    if '1' in SR_METHOD or config.USE_API_BIN:
        aja_matrix = np.loadtxt(config.APK_ajamat_name, dtype = 'int')
        api_feature_with_label = pd.read_csv('{}{}'.format(BASE_PATH, config.API_feature_name))
        std_apis = GetAPIList(api_feature_with_label)
    ###########################

    # ##########################
    # Normalize the feature dataset
    # ##########################
    
    # Separating samples and labels
    for column_to_be_removed in ['mw_name', 'mw_name_INT', 'mw_family', 'Unnamed: 0']:
        if column_to_be_removed in samples:
            print('Found column to remove: removing {}'.format(column_to_be_removed))
            samples.drop(column_to_be_removed,axis=1,inplace=True)

    if 'mw_family_INT' in samples: 
        samples.rename(columns={"mw_family_INT": "label"}, inplace = True)
    samples_without_label = samples.drop('label',axis=1)
    samples_without_label = samples_without_label.astype(np.float32)
    samples_labels = samples['label']

    if np.any(samples_labels > 1):
        print("labels should be 0 or 1")
        exit()

    if config.USE_API_BIN and std_apis[0] in samples_without_label:
        #  use binary API feature
        print('converting API features to binary...')
        API_features = np.array(samples_without_label[std_apis])
        API_features[np.nonzero(API_features)] = 1
        print('API features shape ', API_features.shape)
        samples_without_label[std_apis] = API_features
        print('#'*20,'DONE','#'*20)

    # ###############
    # LABELS
    # ###############
    labels_to_consider = ['all']

    # if USE_BINARY_CLASSIFICATION:
    for l_to_consider in labels_to_consider:

        if SHOW_DEBUG:
            print('-'*50)
            print('>> Considering labels: {}'.format(l_to_consider))
            print('-'*50)

        count_it = 0

        for it in range(NITERATIONS):

            for algorithm_name, algorithm_fun in CLASSIFIERS_TO_RUN.items():
                if SHOW_DEBUG:
                    print('*'*30)
                    print('Running {} on {}...'.format(algorithm_name,CURRENT_DATASET))
                    print('*'*30)

                for kfold_name in KFOLDS_TO_RUN:

                    # Extending the labels depending whether I'm using binary classification
                    labels_in_current_iteration = None
                    if l_to_consider == 'all':
                        labels_in_current_iteration = getLabels(samples_labels, None)
                    else:
                        labels_in_current_iteration = getLabels(samples_labels, l_to_consider)

                    skf = None

                    if kfold_name == "Stratified":
                        skf = sklearn.cross_validation.StratifiedKFold(
                                    labels_in_current_iteration,
                                    n_folds=NFOLDS,
                                    shuffle=True,
                                    random_state=0) # for deterministic shuffling
                    elif kfold_name == "Simple":
                        skf = KFold(
                                    n_splits=NFOLDS,
                                    shuffle=True,
                                    random_state=0).split(samples_without_label)
                    else:
                        raise Exception("The following K-fold is not supported: {}".format(kfold_name))

                    for method_flag in SR_METHOD:
                        count = 0
                        count_it = 0        
                        y_true_total = []
                        y_pred_total = []
                        y_prob_total = []

                        frames = []

                        if samples_without_label.shape[1] < 2 and method_flag == '0':
                            continue

                        print('*'*50)
                        print('method_flag', method_flag)

                        for train_index, test_index in skf:
                            count += 1
                            count_it += 1
                            X_train, X_test = samples_without_label.iloc[train_index], samples_without_label.iloc[test_index]
                            y_train, y_true = labels_in_current_iteration.iloc[train_index], labels_in_current_iteration.iloc[test_index]
                            #################################################
                            # code for adding sus and sr features dynamically
                            train_sr_feature = test_sr_feature = pd.DataFrame()
                            if method_flag != '0':
                                API_ftrs_for_train = api_feature_with_label.iloc[train_index]
                                if 'mw_name' in API_ftrs_for_train:
                                    API_ftrs_for_train.drop('mw_name', axis = 1, inplace = True)
                                sr_feature = GenAllSR(API_ftrs_for_train, api_feature_with_label, aja_matrix, std_apis, config.group_width)

                                train_sr_feature = sr_feature.iloc[train_index]
                                test_sr_feature = sr_feature.iloc[test_index]
                                X_train = pd.concat([X_train, train_sr_feature], axis = 1)
                                X_test = pd.concat([X_test, test_sr_feature], axis = 1)

                                if 'zero' in X_train:
                                    X_train.drop('zero', axis = 1, inplace = True)
                                    X_test.drop('zero', axis = 1, inplace = True)
                            #################################################


                            ''' append sus/sr samples for figures '''
                            cv_samples = X_test.join(y_true)
                            frames.append(cv_samples)

                            y_train = list(y_train)
                            y_true = list(y_true)
                            
                            if algorithm_name == 'DT':
                                y_pred, y_prob, feature_importances, clf = algorithm_fun(X_train.as_matrix(), y_train, X_test.as_matrix())
                                
                            else:
                                y_pred, y_prob, feature_importances = algorithm_fun(X_train.as_matrix(), y_train, X_test.as_matrix())

                            if len(y_true_total):
                                y_true_total = np.concatenate((y_true_total, y_true), axis = 0)
                                y_pred_total = np.concatenate((y_pred_total, y_pred), axis = 0)
                                y_prob_total = np.concatenate((y_prob_total, y_prob), axis = 0)
                            else:
                                y_true_total = y_true
                                y_pred_total = y_pred
                                y_prob_total = y_prob

                            dataset_name = CURRENT_DATASET
                            if "_ety" in CURRENT_DATASET:
                                CURRENT_DATASET= CURRENT_DATASET.replace("_ety",'')
                            if '0' != method_flag:
                                dataset_name = '{}_sr'.format(CURRENT_DATASET)
                                # dataset_name = '{}_sr_{}'.format(CURRENT_DATASET,method_flag)
                            # DRAWING DECISION_TREE
                            if algorithm_name == 'DT':
                                print("Drawing DT...")                
                                tree_name = "{}/{}_tree_{:02d}".format(RESULTS_FOLDER,dataset_name, count)

                                tree.export_graphviz(clf,
                                                     out_file="{}.dot".format(tree_name),
                                                     feature_names=X_train.columns[:],
                                                     label='all',
                                                     impurity=False,
                                                     max_depth=5,
                                                     filled=True,
                                                     leaves_parallel=True,
                                                     class_names=['BANKER', 'GOODWARE']) # class_names=['BANKER', 'OTHER']
                                for F in ['png', 'ps']:
                                    call(["dot", "-T{}".format(F), "{}.dot".format(tree_name), "-o", "{}.{}".format(tree_name,F)])

                                tree.export_graphviz(clf,
                                                     out_file="{}_maxdepth03.dot".format(tree_name),
                                                     feature_names=X_train.columns[:],
                                                     label='all',
                                                     impurity=False,
                                                     max_depth=3,
                                                     filled=True,
                                                     leaves_parallel=True,
                                                     class_names=['BANKER', 'GOODWARE']) # class_names=['BANKER', 'OTHER']
                                for F in ['png', 'ps']:
                                    call(["dot", "-T{}".format(F), "{}_maxdepth03.dot".format(tree_name), "-o", "{}_maxdepth03.{}".format(tree_name,F)])

                            if SHOW_DEBUG:
                                print('Samples not predicted: {}'.format(set(y_true) - set(y_pred)))
                                print('\tPrecision:{} - Recall:{} - AUC:{}'.format(sklearn.metrics.precision_score(y_true,y_pred, average='binary'),\
                                    sklearn.metrics.recall_score(y_true,y_pred, average='binary'),\
                                    sklearn.metrics.roc_auc_score(y_true,y_prob)))
                            # I am saving feature importance (right now, only for random forest)
                            if feature_importances is not None and SAVE_FEATURE_IMPORTANCES:
                                current_feature_importance = {}
                                current_feature_importance['dataset'] = dataset_name
                                current_feature_importance['n_iteration'] = '{:02d}'.format(count_it)
                                current_feature_importance['classifier'] = algorithm_name
                                current_feature_importance['considered_class'] = l_to_consider
                                for i, col in enumerate(X_train.columns):
                                    current_feature_importance[col] = feature_importances[i]
                                feature_importance_matrix = feature_importance_matrix.append(current_feature_importance,ignore_index=True)
                            else:
                                pass

                            # ##########################
                            # Confusion matrix
                            # ##########################

                            if l_to_consider == 'all':

                                if SAVE_CONFUSION_MATRIX:
                                    confusion_matrix =  sklearn.metrics.confusion_matrix(np.array(y_true), np.array(y_pred))
                                    cm = pd.DataFrame(confusion_matrix)
                                    df_sum = cm.sum(axis=1).astype('float')
                                    cm_normalized = cm.divide(df_sum.astype('float'),axis=0)
                                    cm_normalized.to_csv( RESULTS_FOLDER + 'cm_percentage_{}_{}_{:02d}.csv'.format(algorithm_name, dataset_name,count_it),index=False)

                            # ##########################
                            # Computing performance
                            # ##########################

                            # If I am considering a multi-class problem
                            if l_to_consider == 'all':

                                y_pred = list(y_pred)
                                y_true = list(y_true)


                                if algorithm_name == 'RF' and SAVE_PROBABILITY_MATRIX:
                                    print(">> Computing probability matrix of class belongness with RF...")
                                    y_pred_proba, fi_proba = run_random_forest_v2(X_train.as_matrix(), y_train, X_test.as_matrix())
                                    y_pred_proba_df = pd.DataFrame(y_pred_proba)
                                    y_pred_proba_max = y_pred_proba_df.max(axis=1)
                                    y_pred_proba_idxmax = y_pred_proba_df.idxmax(axis=1)

                                # saving performance matrix
                                L_prediction = len(y_pred)
                                for idx_prediction in range(L_prediction):
                                    # updating the numpy arrays
                                    prediction_matrix['idx'] = np.append(prediction_matrix['idx'], test_index[idx_prediction])
                                    prediction_matrix['y_true'] = np.append(prediction_matrix['y_true'], y_true[idx_prediction])
                                    prediction_matrix['y_pred'] = np.append(prediction_matrix['y_pred'], y_pred[idx_prediction])
                                    prediction_matrix['classifier'] = np.append(prediction_matrix['classifier'], algorithm_name)
                                    prediction_matrix['cross_val_iteration_number'] = np.append(prediction_matrix['cross_val_iteration_number'], int(count))
                                    prediction_matrix['dataset'] = np.append(prediction_matrix['dataset'], dataset_name)
                                    prediction_matrix['n_iteration'] = np.append(prediction_matrix['n_iteration'], count_it)
                                    prediction_matrix['considered_class'] = np.append(prediction_matrix['considered_class'], l_to_consider)
                                    if algorithm_name == 'RF' and SAVE_PROBABILITY_MATRIX:
                                        prediction_matrix['y_pred_proba_max'] = np.append(prediction_matrix['y_pred_proba_max'],y_pred_proba_max[idx_prediction])
                                        prediction_matrix['y_pred_proba_idxmax'] = np.append(prediction_matrix['y_pred_proba_idxmax'],y_pred_proba_idxmax[idx_prediction])
                                    else:
                                        prediction_matrix['y_pred_proba_max'] = np.append(prediction_matrix['y_pred_proba_max'],-1)
                                        prediction_matrix['y_pred_proba_idxmax'] = np.append(prediction_matrix['y_pred_proba_idxmax'],-1)

                                recall = None
                                precision = None
                                fscore = None

                            # ######################################
                            # BINARY CLASSIFICATION PERFORMANCE
                            # ######################################
                            else:

                                if SHOW_DEBUG:
                                    print(">> Considering binary classification problem with RF: label {}.".format(l_to_consider))

                                y_pred = list(y_pred)
                                y_true = list(y_true)

                                # MACRO
                                (precision_avg, recall_avg) = compute_macro_stats(y_pred,y_true)
                                MaRecall_custom = recall_avg
                                MaF_custom = compute_fscore(MaPrecision_custom,MaRecall_custom)

                                # MICRO
                                (precision_avg, recall_avg) = compute_micro_stats(y_pred,y_true)
                                MiPrecision_custom = precision_avg
                                MiRecall_custom = recall_avg
                                MiF_custom = compute_fscore(MiPrecision_custom,MiRecall_custom)

                                # Class size
                                class_size = 0
                                for lab in samples_labels:
                                   if lab == l_to_consider:
                                       class_size += 1

                                MiAUC = 0
                                MaAUC = 0

                                try:
                                    MiAUC = sklearn.metrics.roc_auc_score(y_true,y_pred,average='micro')
                                    MaAUC = sklearn.metrics.roc_auc_score(y_true,y_pred,average='macro')
                                except Exception as e:
                                    raise Exception('Error while computing MiAUC/MaAUC:', e)

                                print('MiAUC: {} - MaAUC: {}'.format(MiAUC, MaAUC))

                        cm_total_norm, result = GetResultRow(y_true_total, y_pred_total, y_prob_total, dataset_name, algorithm_name, l_to_consider, kfold_name, len(samples))
                        cm_total_norm.transpose().to_csv(RESULTS_FOLDER + 'CM_{}_{}.csv'.format(algorithm_name, dataset_name,index=False))
                        final_results_mat=final_results_mat.append(result, ignore_index = True)
                        # save intermediate results
                        final_results_mat.to_csv(RESULTS_FOLDER + 'Inter_Result_{}.csv'.format(time.time()))

                if SAVE_FEATURE_IMPORTANCES:
                    feature_importance_matrix.to_csv(RESULTS_FOLDER + 'feature_importance_INCOMPLETE.csv'.format(CURRENT_DATASET),index=False)

                # Saving results full
                prediction_matrix_df = pd.DataFrame.from_dict(prediction_matrix)
                prediction_matrix_df.to_csv(RESULTS_FOLDER + 'y_pred_incomplete.csv',index=False)


''' output sus/sr samples'''
X_test_output = pd.concat(frames)
X_test_output.to_csv("output_o.csv")

# Saving feature importance
if SAVE_FEATURE_IMPORTANCES:
    feature_importance_matrix.to_csv(RESULTS_FOLDER + 'feature_importance.csv')

prediction_matrix_df = pd.DataFrame.from_dict(prediction_matrix)
final_results_mat.to_csv(RESULTS_FOLDER + 'Final_Result_Mat.csv',index=False)

print('#'*30)
print('Classification script: completed')
print('#'*30)
