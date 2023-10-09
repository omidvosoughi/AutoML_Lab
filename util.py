from functools import partial
from typing import List, Optional
import numpy as np
from tqdm import tqdm 

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import NumericalHyperparameter, Constant, CategoricalHyperparameter, \
    OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters, get_one_exchange_neighbourhood
from sklearn.impute import SimpleImputer


def convert_configurations_to_array(configs: List[Configuration]):
    return np.array([config.get_array() for config in configs], dtype=np.float64)


def get_num_available_hp_values(cs: ConfigurationSpace) -> List[int]:
    """
    get the number of the available hyperparameter values for the configuration space
    """
    sample = cs.sample_configuration()
    hps = cs.get_hyperparameters()
    num_available_hp_values = [0] * len(hps)
    for i, hp in enumerate(hps):
        num_neighbour = hp.get_num_neighbors(sample)
        num_available_hp_values[i] = num_neighbour + 1
    return num_available_hp_values


def transform_continuous_designs(design: np.ndarray,
                                 cs: ConfigurationSpace) -> List[Configuration]:
    """
    transform the array design from [0,1]^N to a list of configurations
    """
    params = cs.get_hyperparameters()
    
    for idx, param in enumerate(params):
        if isinstance(param, NumericalHyperparameter):
            continue
        elif isinstance(param, Constant):
            raise RuntimeError("Might have issues here.")
            # add a vector with zeros
            design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
            design_[:, :idx] = design[:, :idx]
            design_[:, idx + 1:] = design[:, idx:]
            design = design_
        elif isinstance(param, CategoricalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10 ** -10
            design[:, idx] = np.array(v_design * len(param.choices), dtype=np.int)
        elif isinstance(param, OrdinalHyperparameter):
            v_design = design[:, idx]
            v_design[v_design == 1] = 1 - 10 ** -10
            design[:, idx] = np.array(v_design * len(param.sequence), dtype=np.int)
        else:
            raise ValueError("Hyperparameter not supported in LHD")
   
        
    configs = []
    for vector in design:
        conf = deactivate_inactive_hyperparameters(configuration=None,
                                                   configuration_space=cs,
                                                   vector=vector)
        configs.append(conf)
    return configs

'''
def impute_nan(X:np.ndarray):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    return imp.fit_transform(X)
'''

def impute_nan(X:np.ndarray):
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1e10)
    return imp.fit_transform(X)


class RunHistory(object):
    def __init__(self, configurations: Optional[List[Configuration]] = None, costs: Optional[List[float]]=None, full_dataset=True):
        if configurations is None or costs is  None:
            self.rh = []
        elif len(configurations) != len(costs):
            raise ValueError("the length of configurations and costs need to be the same!")
        else:
            self.rh = [(config, cost) for config, cost in zip(configurations, costs)]
        
        self.full_dataset = full_dataset

    def add_items(self,
                  configuration: Configuration,
                  cost: float,
                  elapsed_time: float):
        self.rh.append((configuration, cost, elapsed_time))
        self.rh = sorted(self.rh, key=lambda x: x[1])

    def get_all_configs(self):
        configs = [config_and_cost[0] for config_and_cost in self.rh]
        return configs

    def get_configuration_arrays(self):
        return np.array(
            [config_and_cost[0].get_array() for config_and_cost in self.rh], dtype=np.float64)

    def get_array_and_costs(self):
        configs_and_costs = [list(value) for value in zip(*self.rh)]
        X = convert_configurations_to_array(configs_and_costs[0])
        y = np.array(configs_and_costs[1], dtype=np.float64)

        X = impute_nan(X)

        return X, y

    def get_best_config(self):
        return self.get_all_configs()[0]

    def empty(self):
        return len(self.rh) != 0


class PlotRunHistory():
    def __init__(self, run_histories):
        self.config_data = {}
        self.subset_train_cost_data = {}
        self.train_cost_data = {}
        self.plot_data = {}

        # Combine all run_histories
        for run_history in run_histories:
            full_dataset = run_history.full_dataset
            for (config, cost, elapsed_time) in run_history.rh:
                self.config_data[elapsed_time] = config

                if full_dataset:
                    self.train_cost_data[elapsed_time] = self._change_cost(cost)
                else:
                    self.subset_train_cost_data[elapsed_time] = self._change_cost(cost)
                    self.train_cost_data[elapsed_time] = None

    def evaluate_missing_costs(self, eval_train, eval_test, end_time):
        cost_data = self.train_cost_data.copy()
        print("Evaluate missing costs")
        for time, cost in tqdm(cost_data.items()):
            if cost is None:
                config = self.config_data[time]
                self.train_cost_data[time] = self._change_cost(eval_train(config))

        # Sort first
        self.train_cost_data = data = dict(sorted(self.train_cost_data.items()))

        # We need same x axis to average curves later
        best_key = None
        best_value = 1
        for i in np.arange(0, end_time+1, step=0.2):

            closest_key = min(data.keys(), key=lambda k: abs(k-i))
            corresponding_value = data[closest_key]

            if corresponding_value < best_value:
                best_key = closest_key
                best_value = corresponding_value

            # Only take the best value for plot data
            self.plot_data[i] = best_value

        self.best_train_score = best_value
        self.best_test_score = abs(eval_test(self.config_data[closest_key]))

    def _change_cost(self, cost):
        return 1+cost
        
    def get_plot_data(self):
        return self.plot_data

    def get_best_train_score(self):
        try:
            return self.best_train_score
        except:
            raise RuntimeError("Please run evaluate_missing_costs first.")

    def get_best_test_score(self):
        try:
            return self.best_test_score
        except:
            raise RuntimeError("Please run evaluate_missing_costs first.")

    def get_num_configs(self):
        return len(list(self.train_cost_data.values()))



