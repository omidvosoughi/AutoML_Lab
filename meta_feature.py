from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd
#from src.util import impute_nan
#from scipy.stats import skew
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis, entropy, moment, median_abs_deviation, iqr, gmean, hmean
from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

from typing import Dict, List

import numpy as np
import pandas as pd
import math

from ConfigSpace import Configuration
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

import openml

class MetaFeatures():
    """
    a class that ensembles all the meta feature compute methods
    """
    def __init__(self):
        self.functions = OrderedDict()

    def add_item(self, name):
        def wrapper(func):
            self.functions.update({name: func})
            return func
        return wrapper

metafeatures = MetaFeatures()


#@metafeatures.add_item("NumberOfInstances")
def number_of_instances(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of instance in the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    """

    return X.shape[0]


#@metafeatures.add_item("NumberOfFeatures")
def number_of_features(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of features in the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    """
    
    return X.shape[1]


#@metafeatures.add_item("SkewnessMean")
def skewness_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None
    
    return np.mean(skew(X_copy))


#@metafeatures.add_item("KurtosisMean")
def kurtosis_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    Kurtosis of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    #X_copy = np.delete(X, np.array(categorical), 1)
    #if X_copy.shape[1] == 0:
    #    return None
    
    return np.nanmean(kurtosis(X))


#@metafeatures.add_item("FifthMomentMean")
def fifth_moment_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None

    return np.mean(moment(a=X_copy, moment=5))



#@metafeatures.add_item("MADMean")
def mad_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None
    
    return np.mean(median_abs_deviation(X_copy))


#@metafeatures.add_item("IQRMean")
def iqr_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None
    
    return np.mean(iqr(X_copy))


#@metafeatures.add_item("GeometricMean")
def geometric_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    #X_copy = np.delete(X, np.array(categorical), 1)
    #if X_copy.shape[1] == 0:
    #    return None
    
    return np.nanmean(gmean(X))

#@metafeatures.add_item("Mean")
def mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """


    
    return np.mean(X)


#@metafeatures.add_item("HarmonicMean")
def harmonic_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    skewness of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean value of the skewness
    """

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None
    
    return np.mean(hmean(X_copy))



@metafeatures.add_item("distribution")
def dist(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of nodes of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: 
    """

    X_copy = X.copy()
    X_copy = np.transpose(X_copy)

    # For each feature we calculate statistics
    statistics = []
    for ele in X_copy:
        min_value = np.min(ele)
        max_value = np.max(ele)

        if (max_value - min_value) != 0:
            ele = (ele - min_value) / (max_value - min_value)

        statistics.append(np.std(ele))#(np.std(ele), np.mean(ele)))
        #statistics.append(np.mean(ele))

    # measure euclidian distance
    return statistics


# calculate euclidian distance of cluster centers as a feature:
#@metafeatures.add_item("Cluster_CentersMean2")
def cluster_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of nodes of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: 
    """

    #if y.any() < 1 or y.any() > 0:
    #    return None

    # Fit kmeans and compute cluster centers
    # with k = 2 only for binary classification datasets meaningful
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    # measure euclidian distance
    return np.mean(kmeans.cluster_centers_[1, :])

#@metafeatures.add_item("Cluster_CentersMean")
def cluster_mean(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of nodes of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: 
    """

    #if y.any() < 1 or y.any() > 0:
    #    return None

    # Fit kmeans and compute cluster centers
    # with k = 2 only for binary classification datasets meaningful
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)


    # measure euclidian distance
    return np.mean(kmeans.cluster_centers_[0, :])


#@metafeatures.add_item("ClassEntropy")
def class_entropy(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    class entropy of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: class entropy
    """

    value, counts = np.unique(y, return_counts=True)
    result = entropy(counts, base=2)

    return result


#@metafeatures.add_item("NumberOfNodes")
def number_of_nodes(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of nodes of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: number of nodes of decision tree
    """

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    return tree.tree_.node_count


#@metafeatures.add_item("Depth")
def depth(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    depth of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: number of nodes of decision tree
    """

    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    
    return tree.get_depth()


#@metafeatures.add_item("NumberOfLeaves")
def number_of_leaves(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    number of leaves of a decision tree
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: number of leaves of decision tree
    """

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    return tree.get_n_leaves()


def pca(X: np.ndarray, y: np.ndarray, categorical: np.ndarray, fraction=0.95):
    """
    fraction of dimensions that explains the 95% of the variances
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: fraction of 95% PCA features
    """
    assert fraction <= 1

    X_copy = np.delete(X, np.array(categorical), 1)
    if X_copy.shape[1] == 0:
        return None
    
    pca = decomposition.PCA()
    pca.fit(X_copy)

    accumulated = 0
    count = 0
    for var in pca.explained_variance_ratio_:
        accumulated += var
        count += 1
        if accumulated > fraction:
            return 1/count


#@metafeatures.add_item("PCA95")
def pca_95(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    fraction of dimensions that explains the 95% of the variances
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: fraction of 95% PCA features
    """

    X_copy = X.copy()
    X_copy = np.transpose(X_copy)

    # For each feature we calculate statistics
    statistics = []
    for i, ele in enumerate(X_copy.copy()):
        min_value = np.min(ele)
        max_value = np.max(ele)

        if (max_value - min_value) != 0:
            ele = (ele - min_value) / (max_value - min_value)

        #statistics.append(np.std(ele))#(np.std(ele), np.mean(ele)))
        #statistics.append(np.mean(ele))
        X_copy[i] = ele

    X_copy = np.transpose(X_copy)

    
    return pca(X_copy, y, categorical, fraction=0.95)


#@metafeatures.add_item("PCA50")
def pca_50(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    fraction of dimensions that explains the 50% of the variances
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: fraction of 95% PCA features
    """
    
    return pca(X, y, categorical, fraction=0.90)


#@metafeatures.add_item("PCA25")
def pca_25(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    fraction of dimensions that explains the 50% of the variances
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: fraction of 95% PCA features
    """
    
    return pca(X, y, categorical, fraction=0.99)


#@metafeatures.add_item("LandmarkLDA")
def landmark_LDA(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):

    model = LinearDiscriminantAnalysis()
    classifier = OneVsRestClassifier(model).fit(X, y)

    return np.mean(cross_val_score(classifier, X, y, cv=5, scoring="accuracy"))
    

#@metafeatures.add_item("LandmarkStump")
def landmark_stump(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    """
    mean accuracy of the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return: mean accuracy of the dataset
    """
    
    model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    classifier = OneVsRestClassifier(model).fit(X, y)

    return np.mean(cross_val_score(classifier, X, y, cv=5, scoring="accuracy"))



#@metafeatures.add_item("Landmark1NN")
def landmark_1NN(X: np.ndarray, y: np.ndarray, categorical: np.ndarray):
    model = KNeighborsClassifier(n_neighbors=1)
    classifier = OneVsRestClassifier(model).fit(X, y)

    return np.mean(cross_val_score(classifier, X, y, cv=5, scoring="accuracy"))





def build_meta_features(X: np.ndarray, y: np.ndarray) -> pd.Series:
    """
    build meta feature vector for the dataset
    :param X: input feature matrix
    :param y: target label
    :param categorical: a vector indicating if the feature is numerical or categorical,
    should have the same shape as X.shape[1]
    :return:
    """



    categorical = np.array([False for _ in range(X.shape[1])])

    meta_functions = metafeatures.functions
    func_keys = meta_functions.keys()

    # Compute meta feature vector here
    func_values = {}
    for key, func in meta_functions.items():
        func_values[key] = func(X, y, categorical)

    return func_values


def k_nearest_datasets(meta_features, # Pre-calculated datasets
                       reference_meta_features, # Dataset which is to be evaluated
                       k: int = 5,
                       metric: str = "l2") -> np.ndarray:

    # Find the indices of the datasets that are closest to meta_feature_test
    #assert len(meta_features_reference) == 1

    result = {}
    for index, value in meta_features.items():
        
        #not_nan_array = [not (math.isnan(e1) or math.isnan(e2)) for e1, e2 in zip(test_vector_copy, train_vector)]
        
        # Remove values with nan 
        # So we don't run into problems while measuring the distance
        #train_vector = train_vector[not_nan_array]
        #test_vector_copy = test_vector_copy[not_nan_array]



        if metric == "l2":
            d = distance.euclidean(list(reference_meta_features.values()), list(value.values()))
        else:
            raise RuntimeError("Not implemented")
        
        result[index] = d
    
    # Sort for shortest distances
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}

    k = min(k, len(list(result.keys())))
    best = list(result.keys())[:k]
    
    return np.array(best), result
