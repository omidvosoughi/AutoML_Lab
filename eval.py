import time
from pathlib import Path
import typing
import copy

import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

from ConfigSpace import Configuration, ConfigurationSpace
from hpobench.util.rng_helper import get_rng
from util import RunHistory

MAX_TRAINING_TIME = 1200
MODEL_DICT = {
    "kNN": KNeighborsClassifier,
    "RF": RandomForestClassifier,
    "ada": AdaBoostClassifier,
    "linear": LogisticRegression,
    "gb": GradientBoostingClassifier,
    "ET": ExtraTreesClassifier,
    "SVM": SVC
}


def load_data(dataset_name):
    features_train_dir = dataset_name + "_features_train.data"
    labels_train_dir = dataset_name + "_labels_train.data"

    features_test_dir = dataset_name + "_features_test.data"
    labels_test_dir = dataset_name + "_labels_test.data"

    X_train = pd.read_csv(str(features_train_dir), header=None, delim_whitespace=True).to_numpy()
    y_train = pd.read_csv(str(labels_train_dir), header=None, delim_whitespace=True).to_numpy().squeeze()
    X_test = pd.read_csv(str(features_test_dir), header=None, delim_whitespace=True).to_numpy()
    y_test = pd.read_csv(str(labels_test_dir), header=None, delim_whitespace=True).to_numpy().squeeze()

    return X_train, y_train, X_test, y_test



def select_data(X_train, y_train, groups=None, max_groups=10, shuffle=True):

    rng = get_rng(0)

    train_size = np.shape(X_train)[0]
    train_index = np.arange(train_size)

    if shuffle:
        rng.shuffle(train_index)

    if isinstance(groups, int):
        groups = [groups]

    if (isinstance(groups, list) or isinstance(groups, tuple)) and len(groups) > 0:
        group_borders = [int(i/max_groups*len(train_index)) for i in range(max_groups+1)]

        selected_ids = []
        for group in groups:
            selected_ids += list(range(group_borders[group], group_borders[group+1]))
        
        selected_ids = sorted(selected_ids)
        train_idx = train_index[selected_ids]
    else:
        train_idx = train_index

    X = X_train[train_idx]
    y = y_train[train_idx]

    return X, y


def evaluate_training_data(configuration: Configuration,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           group=None,
                           max_groups=128,
                           shuffle=True):

    classifier_kwargs = copy.deepcopy(configuration.get_dictionary())
    classifier_name = classifier_kwargs["classifiers_name"]
    del classifier_kwargs["classifiers_name"]
    classifier_kwargs_modified = {}
    for key in classifier_kwargs.keys():
        new_key = key.split(':')[1]
        classifier_kwargs_modified[new_key] = classifier_kwargs[key]

    classifier = MODEL_DICT[classifier_name](**classifier_kwargs_modified)

    X, y = select_data(X_train, y_train, group, max_groups, shuffle)

    accuracy_scorer = make_scorer(balanced_accuracy_score)

    # or you can use whatever loss you like here
    S = cross_val_score(classifier, X, y, scoring=accuracy_scorer, cv=3)

    # We want to maximize accuracy
    return -np.mean(S)


def evaluate_test_data(configuration: Configuration,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       x_test: np.ndarray,
                       y_test: np.ndarray):
    """
    train the model on training set and test it on test set for the final score
    :param configuration: configuration for the model
    :param X_train: training X
    :param y_train: training y
    :param x_test: test X
    :param y_test: test y
    :return: score: float
    """

    classifier_kwargs = copy.deepcopy(configuration.get_dictionary())
    classifier_name = classifier_kwargs["classifiers_name"]
    del classifier_kwargs["classifiers_name"]
    classifier_kwargs_modified = {}
    for key in classifier_kwargs.keys():
        new_key = key.split(':')[1]
        classifier_kwargs_modified[new_key] = classifier_kwargs[key]

    classifier = MODEL_DICT[classifier_name](**classifier_kwargs_modified)

    classifier.fit(X_train, y_train)

    accuracy_scorer = make_scorer(balanced_accuracy_score)
    test_score = accuracy_scorer(classifier, x_test, y_test)
    
    # We want to maximize accuracy
    return -test_score

