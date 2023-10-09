import os
import time
from pathlib import Path
import typing
import json
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

from ConfigSpace import Configuration, ConfigurationSpace
#from hpolib.util.rng_helper import get_rng

from sklearn_configspace import build_configuration_space

from surrogate_models import GaussianProcess, RandomForest, BayesianNeuralNetworkDropOut
from acquisition_functions import EI
from optimizers import *
from meta_feature import build_meta_features, k_nearest_datasets
import itertools

from plot import plot_subsets, plot_thresholds, create_dirs
from table import write_thresholds

from util import RunHistory, PlotRunHistory
from eval import evaluate_training_data, load_data, evaluate_test_data, select_data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="labAML")
    #parser.add_argument('--dataset_name', type=str, default='sylvine')

    return parser.parse_args()


def find_best_subset(dataset_name, k=128, threshold=0.5, best=True):
    assert threshold >= 0.1

    # First create k groups
    groups = [tuple([group]) for group in range(0, k)]

    # Get reference features
    X_train, y_train, X_test, y_test = load_data(os.path.join("datasets", dataset_name, dataset_name))
    reference_meta_features = build_meta_features(X_train, y_train)

    # And get meta features for every group/subset
    meta_features = {}
    for group in groups:
        X_group, y_group = select_data(X_train, y_train, group, max_groups=k)
        meta_features[group] = build_meta_features(X_group, y_group)

    # Sort them
    best_groups, _ = k_nearest_datasets(meta_features, reference_meta_features, k=k)

    plot_data = {}
    intermediate_group = [] 

    if not best:
        best_groups = best_groups[::-1]

    for group in best_groups:
        intermediate_group += list(group.copy())

        X_group, y_group = select_data(X_train, y_train, intermediate_group, k)
        features = {
            tuple(intermediate_group): build_meta_features(X_group, y_group)
        }

        x = float(len(intermediate_group)/k) # Relative size of sub dataset
        _, results = k_nearest_datasets(
            features, 
            reference_meta_features, 
            k=1
        )

        plot_data[tuple(intermediate_group)] = (x, results[tuple(intermediate_group)])

    group_keys = list(plot_data.keys())
    choosen_group_index = int(len(list(plot_data.keys())) * threshold) - 1
    choosen_group = group_keys[choosen_group_index]

    '''
    start_value = None
    for group, (_, y) in plot_data.items():
        if start_value is None:
            start_value = y
                    
        if y < start_value * threshold:
            best_group = group
            break
    '''
        
    return choosen_group, plot_data


def optimize(X_train, 
             y_train, 
             rh, 
             model,
             acq_func,
             acq_optimizer,
             acq_optimizer_kwargs,
             group,
             start_time,
             end_time):

    while time.time() - start_time < end_time:
        X, y = rh.get_array_and_costs()

        model.train(X, y)
        acq_func.update(model=model, eta=np.min(y)) 

        next_config = next(acq_optimizer.maximize(rh, **acq_optimizer_kwargs))
        loss = evaluate_training_data(next_config, X_train, y_train, group)

        rh.add_items(next_config, loss, time.time() - start_time)


def hyperparameter_optimization(cs: ConfigurationSpace,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                group,
                                start_time,
                                end_time,
                                random_search=False,
                                seed=0) -> Configuration:

    # Define models for hyperparameter search
    model = GaussianProcess()
    acq_func = EI(model)
    optimizer_kwargs = {"num_points": 100}
    acq_optimizer = GlobalAndLocalSearch(acquisition_function=acq_func, config_space=cs)
    acq_local_optimizer = LocalSearch(acquisition_function=acq_func, config_space=cs)
    cs.seed(seed)
    init_config = cs.sample_configuration(1)

    if random_search:
        group = None

    rh = RunHistory(full_dataset=random_search)
    # Fill run history with first config

    rh.add_items(
        init_config,
        evaluate_training_data(init_config, X_train=X_train, y_train=y_train, group=group),
        time.time() - start_time
    )

    if random_search:
        print("[+] Start random search")
        optimize(
            X_train=X_train,
            y_train=y_train,
            rh=rh,
            model=model,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            acq_optimizer_kwargs=optimizer_kwargs,
            group=group,
            start_time=start_time,
            end_time=end_time,
        )

        return rh.get_best_config(), [rh]
    else:
        print("[+] Start search on sub dataset")
        optimize(
            X_train=X_train,
            y_train=y_train,
            rh=rh,
            model=model,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            acq_optimizer_kwargs=optimizer_kwargs,
            group=group, # This is important
            start_time=start_time,
            end_time=end_time * 3/4,
        )

        # Get the best configuration from rh
        print("[+] Get best configuration from sub dataset and do evaluation on full dataset")
        best_config = rh.get_best_config()
        loss = evaluate_training_data(best_config, X_train=X_train, y_train=y_train, group=None)

        # Add to new runhistory
        rh_full_dataset = RunHistory(full_dataset=True)
        rh_full_dataset.add_items(best_config, loss, time.time() - start_time)

        # Perform local search
        print("[+] Start local search on full dataset")
        optimize(
            X_train=X_train,
            y_train=y_train,
            rh=rh_full_dataset,
            model=model,
            acq_func=acq_func,
            acq_optimizer=acq_local_optimizer, # Use local optimizer here
            acq_optimizer_kwargs=optimizer_kwargs,
            group=None, # This is important
            start_time=start_time,
            end_time=end_time,
        )

        return rh_full_dataset.get_best_config(), [rh, rh_full_dataset]


if __name__ == '__main__':
    args = parse_args()
    #dataset_name = args.dataset_name

    #X_train, y_train, X_test, y_test = load_data(os.path.join("datasets", dataset_name, dataset_name))
    # Define all the classifiers as hyperparameters with their underlying hyperparameters
    cs_classifiers = build_configuration_space()

    max_time = 1200
    repeatings = 5
    dataset_names = ["jasmine", "christine"]#, "sylvine", "christine", "madeline"]
    thresholds = ["RS", 0.1, 0.3, 0.5]#, 0.1, 0.3, 0.5] # Choose subset which is x times better than the worst
    debug = False

    # Experiment 1: Show relative subset size over distance
    '''
    print("------ Experiment 1 ------]")
    plot_data = {}
    for dataset_name in dataset_names:
        print(f"[+++] {dataset_name}")
        _, plot_data_best = find_best_subset(dataset_name, best=True)
        _, plot_data_worst = find_best_subset(dataset_name, best=False)
        plot_data[f"{dataset_name} (best)"] = plot_data_best
        plot_data[f"{dataset_name} (worst)"] = plot_data_worst
    
    plot_subsets(os.path.join("plots", "datasets.png"), plot_data)
    '''


    # Experiment 2: Show runs with different thresholds
    print("------ Experiment 2 ------]")
    table_data = {}

    for seed in range(repeatings):
        for dataset_name in dataset_names:
            print(f"[+++] {dataset_name}")

            data_filename = os.path.join("data", dataset_name, f"thresholds-{max_time}.p")
            create_dirs(data_filename)

            plot_data = {}
            if os.path.exists(data_filename):
                with open(data_filename, 'rb') as file:
                    plot_data = pickle.load(file)
            
            X_train, y_train, X_test, y_test = load_data(os.path.join("datasets", dataset_name, dataset_name))
            evaluate_full_training_data = lambda config: evaluate_training_data(config, X_train=X_train, y_train=y_train, group=None)
            evaluate_full_test_data = lambda config: evaluate_test_data(config, X_train, y_train, X_test, y_test)

            for threshold in thresholds:
                print(f"{threshold} ({seed+1}/{repeatings})")
            
                if threshold in plot_data:
                    arr = plot_data[threshold]
                    if seed < len(arr):
                        continue

                print(f"Run {seed+1}")
                start_time = time.time()

                # First find best subset (= best group)
                best_group = None
                if threshold != "RS":
                    best_group, _ = find_best_subset(dataset_name, threshold=threshold, best=True)

                # Get all configs with time as key
                best_config, run_histories = hyperparameter_optimization(
                    cs_classifiers, 
                    X_train, 
                    y_train, 
                    best_group, 
                    start_time,
                    max_time,
                    random_search=threshold == "RS",
                    seed=seed
                )

                print(f"Elapsed time after HPO: {time.time() - start_time}")

                plot_run_history = PlotRunHistory(run_histories)
                plot_run_history.evaluate_missing_costs(
                    evaluate_full_training_data,
                    evaluate_full_test_data,
                    max_time
                )

                print(f"Elapsed time after evaluate missing costs: {time.time() - start_time}")
                    
                if threshold not in plot_data:
                    plot_data[threshold] = [plot_run_history]
                else:
                    plot_data[threshold] += [plot_run_history]

                if not debug:
                    with open(data_filename, 'wb') as file:
                        pickle.dump(plot_data, file)
            
        table_data[dataset_name] = plot_data
            
        # Plot thresholds
        plot_thresholds(os.path.join("plots", dataset_name, f"thresholds-{max_time}.png"), plot_data, max_time * 3/4)
    
    write_thresholds(os.path.join("tables", f"thresholds-{max_time}.csv"), table_data)
    