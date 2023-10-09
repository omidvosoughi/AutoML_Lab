import numpy as np
import collections
from collections import defaultdict
import pandas as pd
from plot import create_dirs


def write_thresholds(filename, data):
    '''
    Goal: Write a pandas table with labels as columns and datasets as rows
    A cell should consist of (
        best_training_accuracy,
        best_training_accuracy_epoch,
        best_test_accuracy,
        best_test_accuracy_epoch,
        how many configs have been evaluated,
        last_time
    )
    or as dict

    {
        'datasets': [A, B, C, D, E],
        'RS': [(0.85, 300, 0.83, 400, 94000, last_time), (), (), (), ()],
        'T=0.5': [(), (), (), (), ()],
    }

    '''
    create_dirs(filename)

    table_data = defaultdict(list)

    last_subset_sample = 0

    for dataset_name, dataset_data in data.items():
        table_data["datasets"].append(dataset_name)

        #table_dataset_data = defaultdict(list)

        # Label is threshold or RS
        for i, (label, plot_run_histories) in enumerate(dataset_data.items()):
            best_train_scores = []
            best_test_scores = []
            num_configs = []

            for plot_run_history in plot_run_histories:
                best_train_scores.append(plot_run_history.get_best_train_score())
                best_test_scores.append(plot_run_history.get_best_test_score())     
                num_configs.append(plot_run_history.get_num_configs())     

            table_data[label].append(
                (
                    np.mean(np.array(best_train_scores)), 
                    np.mean(np.array(best_test_scores)),
                    np.mean(np.array(num_configs)),
                )
            )
    
    df = pd.DataFrame(data=table_data)
    df.to_csv(filename, index=False)

