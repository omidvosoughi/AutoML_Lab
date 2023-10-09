"""
ConfigSpace built for sklearn models
"""
import copy

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, Constant


def build_configuration_space():
    # kNN
    n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 30, default_value=10)
    p = UniformIntegerHyperparameter("p", 1, 4, default_value=2)
    leaf_size = UniformIntegerHyperparameter("leaf_size", 10, 50, default_value=30)

    cs_kNN = ConfigurationSpace()
    cs_kNN.add_hyperparameters([n_neighbors, p, leaf_size])

    # SVM
    c = UniformFloatHyperparameter("C", 1.0, 1e3, log=True, default_value=10.)
    gamma = UniformFloatHyperparameter("gamma", 1e-4, 1e-3, log=True, default_value=1e-3)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True, default_value=1e-3)
    #kernel = Constant("kernel", "rbf")

    cs_SVM = ConfigurationSpace()
    #cs_SVM.add_hyperparameters([c, gamma, tol, kernel])
    cs_SVM.add_hyperparameters([c, gamma, tol])


    # DT
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=10)
    max_features = UniformFloatHyperparameter("max_features", 0.01, 0.99, log=True, default_value=0.1)
    min_samples_split = UniformFloatHyperparameter("min_samples_split", 0.01, 0.99, log=True, default_value=0.1)
    min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 0.01, 0.49, log=True, default_value=0.1)
    min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0.01, 0.49, log=True, default_value=0.1)
    min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.5, default_value=0.25)

    cs_DT = ConfigurationSpace()
    dt_hps = [max_depth, max_features, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease]
    cs_DT.add_hyperparameters(dt_hps)

    # RF
    n_trees = UniformIntegerHyperparameter("n_estimators", 1, 15, default_value=10)
    cs_RF = ConfigurationSpace()
    rf_hps = [copy.copy(dt_hp) for dt_hp in dt_hps]
    rf_hps.append(n_trees)
    cs_RF.add_hyperparameters(rf_hps)

    # ADA
    n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 200, log=False, default_value=50)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-4, 1e1, log=True, default_value=1e-2)

    cs_ADA = ConfigurationSpace()
    cs_ADA.add_hyperparameters([n_estimators, learning_rate])

    # Linear
    c_linear = UniformFloatHyperparameter("C", 1e-2, 1e2, log=True, default_value=1.)
    intercept_scaling = UniformFloatHyperparameter("intercept_scaling", 1e-2, 1e2, log=True, default_value=1.)

    cs_linear = ConfigurationSpace()
    cs_linear.add_hyperparameters([c_linear, intercept_scaling])

    # GB
    learning_rate_gb =  UniformFloatHyperparameter("learning_rate", 1e-5, 1e1, log=True, default_value = 1e-1)
    max_depth_gb = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=3)
    n_estimators_gb = UniformIntegerHyperparameter("n_estimators", 50, 200, default_value = 100)
    subsample_gb = UniformFloatHyperparameter("subsample", 0.5, 1.0, log=False, default_value = 1.0)
    #min_samples_split_gb = UniformFloatHyperparameter("min_samples_split_gb", 0.01, 0.99, log=True, default_value=0.1)
    #min_samples_leaf_gb = UniformFloatHyperparameter("min_samples_leaf_gb", 0.01, 0.49, log=True, default_value=0.1)

    cs_gb = ConfigurationSpace()
    cs_gb.add_hyperparameters([learning_rate_gb, max_depth_gb, n_estimators_gb, subsample_gb])

    # ExtraRandomTrees
    max_depth_et = UniformIntegerHyperparameter("max_depth", 1, 50, default_value=25)
    n_estimators_et = UniformIntegerHyperparameter("n_estimators", 50, 200, default_value = 100)
    #min_samples_split_et = UniformFloatHyperparameter("min_samples_split_et", 0.01, 0.99, log=True, default_value=0.1)
    #min_samples_leaf_et = UniformFloatHyperparameter("min_samples_leaf_et", 0.01, 0.49, log=True, default_value=0.1)

    cs_et = ConfigurationSpace()
    cs_et.add_hyperparameters([max_depth_et, n_estimators_et])

    # GaussianProcessClassifier
    n_restarts_optimizer_gpc = UniformIntegerHyperparameter("n_restarts_optimizer", 0, 10, default_value=0)
    max_iter_predict_gpc = UniformIntegerHyperparameter("max_iter_predict", 50, 250, default_value = 100)

    cs_gpc = ConfigurationSpace()
    cs_gpc.add_hyperparameters([n_restarts_optimizer_gpc, max_iter_predict_gpc])

    cs_all = {
        #"kNN": cs_kNN,
        "SVM": cs_SVM,
        "RF": cs_RF,
        "ada": cs_ADA,
        "gb": cs_gb,
    }


    # We define our classifier as hyperparameter
    classifier_name = CategoricalHyperparameter("classifiers_name", cs_all.keys())

    cs_classifiers = ConfigurationSpace()
    cs_classifiers.add_hyperparameter(classifier_name)
    
    for name, cs in cs_all.items():
        cs_classifiers.add_configuration_space(
            prefix=name,
            configuration_space=cs,
            parent_hyperparameter={
                "parent": classifier_name,
                "value": name
            },
        )

    return cs_classifiers
