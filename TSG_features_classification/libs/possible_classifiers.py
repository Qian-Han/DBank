import sklearn.preprocessing as preprocessing
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
from sklearn import tree
from subprocess import call
from sklearn.grid_search import GridSearchCV

def run_k_nearest_neighbor(X_train, y_train, X_test):
    # default = 5 neighbors
    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)

def run_naive_bayes(X_train, y_train, X_test):
    nb = sklearn.naive_bayes.BernoulliNB()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)


def run_decision_tree_clf(X_train, y_train, X_test):
    dt  = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)

    return (y_pred, y_proba[:,1], dt.feature_importances_, dt)

def run_decision_tree(X_train, y_train, X_test):
    dt  = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)

def run_logistic_regression(X_train, y_train, X_test):

    lr = sklearn.linear_model.LogisticRegression(penalty = 'l1',class_weight = 'balanced') #penalty=p, C=c)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)

def run_lin_svc(X_train, y_train, X_test):
    clf = sklearn.svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    y_pred = clf.predict(X_test)
    return (y_pred, y_score, None)

def run_svm(X_train, y_train, X_test):

    clf = sklearn.svm.SVC(kernel = 'linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
#     feature_importances = list(clf.feature_importances_)
    return (y_pred, y_proba[:,1], None)


def run_deeplearning(X_train, y_train, X_test):
    # BernoulliRBM, which assumes the inputs are either binary values or values between 0 and 1, 
    # each encoding the probability that the specific feature would be turned on. This is a good model for character recognition,
    # where the interest is on which pixels are active and which aren’t.
    min_max_scaler = preprocessing.MinMaxScaler()

    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.fit_transform(X_test)

    logistic = sklearn.linear_model.LogisticRegression()
    # svm_clf = sklearn.svm.SVC()
    rbm = sklearn.neural_network.BernoulliRBM(random_state=0, verbose=True) # n_components=512

    logistic.fit(X_train_minmax, y_train)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    classifier.fit(X_train_minmax, y_train)
    
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)

def run_svm_v2(X_train, y_train, X_test):
    clf = sklearn.svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_proba = clf.predict_proba(X_test)

    return (y_pred, y_proba[:,1], None)


def run_random_forest_v2(X_train, y_train, X_test):
    """
    Runs random forest algorithm.
        X_train = samples without labels used for training
        y_train = labels (from ground truth) for training samples
        X_test  = samples without labels used for testing
    """

    rf = sklearn.ensemble.RandomForestClassifier()
    rf.fit(X_train, y_train)
    feature_importances = list(rf.feature_importances_)
    y_pred = rf.predict_proba(X_test)

    return (y_pred, feature_importances)

def run_random_forest(X_train, y_train, X_test):
    """
    Runs random forest algorithm.
        X_train = samples without labels used for training
        y_train = labels (from ground truth) for training samples
        X_test  = samples without labels used for testing
    """
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_features='sqrt')
    rf.fit(X_train, y_train)
    feature_importances = list(rf.feature_importances_)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
 
    return (y_pred, y_proba[:,1], feature_importances)


# *************************************** New Classifier ***************************************
# Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

def run_adaboost(X_train, y_train, X_test):
    # Create and fit an AdaBoosted decision tree    # max_depth = 1
    bdt = AdaBoostClassifier(DecisionTreeClassifier(),
                         algorithm="SAMME",
                         n_estimators=200)
    bdt.fit(X_train, y_train)
    y_pred = bdt.predict(X_test)
    y_proba = bdt.predict_proba(X_test)
    # print(type(y_pred))
    # print('\n'*3)
#     feature_importances = list(bdt.feature_importances_)
    return (y_pred, y_proba[:,1], None)



# *************************************** New Classifier ***************************************
# GBDT
from sklearn.ensemble import GradientBoostingClassifier

def run_gradient_boosting(X_train, y_train, X_test):
    # Create and fit a GradientBoostingClassifier    # max_depth = 3
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=3, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
#     feature_importances = list(clf.feature_importances_)
    return (y_pred, y_proba[:,1], None)

# *************************************** New Classifier ***************************************
# MLPClassifier
from sklearn.neural_network import MLPClassifier
def run_MLPClassifier(X_train, y_train, X_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
#     feature_importances = list(clf.feature_importances_)
    return (y_pred, y_proba[:,1], None)
# *************************************** New Classifier ***************************************
# Keras NNClassifier

# Create Keras Neural Network Classifier
def run_KerasNNClassifier(X_train, y_train, X_test):

    from keras.models import Sequential
    from keras.layers import Dense

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=10)
    # output the prediction of the input
    y_pred_tmp = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    # round predictions
    y_pred = [round(x[0]) for x in y_pred_tmp]

    return (y_pred, y_prob, None)
