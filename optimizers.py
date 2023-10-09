import abc
import logging
import numpy as np
from typing import List, Union, Tuple, Optional, Iterator
import itertools
from scipy.optimize._shgo_lib.sobol_seq import Sobol

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

from util import RunHistory, get_num_available_hp_values, transform_continuous_designs, get_one_exchange_neighbourhood
from acquisition_functions import AbstractAcquisitionFunction

import itertools
import operator
from scipy.stats import norm

class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.
    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__
        )
        self.acquisition_function = acquisition_function
        self.config_space = config_space

        if rng is None:
            self.logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
        self,
        runhistory: RunHistory,
        **kwargs,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using ``_maximize``.
        Returns
        -------
        iterable
            An iterable consisting of :class:`configspace.Configuration`.
        """
        b = iter([t[1] for t in self._maximize(runhistory, **kwargs)])
        return b

    @abc.abstractmethod
    def _maximize(
            self,
            runhistory: RunHistory,
            num_points: int,
    ) -> List[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.
        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.
        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`configspace.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value
        Parameters
        ----------
        configs : list(Configuration)
        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """
        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind], configs[ind]) for ind in indices[::-1]]


class GlobalOptimizerGrid(AcquisitionFunctionMaximizer):
    def _maximize(
            self,
            runhistory: RunHistory,
            num_points_each_dimension: int,
    ) -> List[Tuple[float, Configuration]]:

        values = None
        for hp in self.config_space.get_hyperparameters():
            space = np.linspace(0, 1, num=num_points_each_dimension)
            if values is None:
                values = np.array(space)
            else:
                values = np.stack((values, space))

        values = np.array(list(itertools.product(*values)))

        t = transform_continuous_designs(values, self.config_space)
        return self._sort_configs_by_acq_value(t)[:10]


class GlobalOptimizerRandom(AcquisitionFunctionMaximizer):
    def _maximize(
            self,
            runhistory: RunHistory,
            num_points: int,
    ) -> List[Tuple[float, Configuration]]:

        configurations = self.config_space.sample_configuration(size=num_points)
        return self._sort_configs_by_acq_value(configurations)[:10]


class GlobalOptimizerSobol(AcquisitionFunctionMaximizer):
    def _maximize(
            self,
            runhistory: RunHistory,
            num_points: int,
    ) -> List[Tuple[float, Configuration]]:

        sobol = Sobol()
        design = sobol.i4_sobol_generate(len(self.config_space.get_hyperparameters()), num_points)
        configs = transform_continuous_designs(
            design,
            self.config_space
        )

        return self._sort_configs_by_acq_value(configs)


class LocalSearch(AcquisitionFunctionMaximizer):
    def _maximize(
            self,
            runhistory: RunHistory,
            num_points: int,
            additional_start_points: Optional[List[Tuple[float, Configuration]]] = None,
    ) -> List[Tuple[float, Configuration]]:
        init_points = self._get_initial_points(num_points=num_points,
                                               runhistory=runhistory,
                                               additional_start_points=additional_start_points)
        configs_acq = self._do_search(init_points)

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        return configs_acq

    def _get_initial_points(
            self,
            num_points: int,
            runhistory: RunHistory,
            additional_start_points: Optional[List[Tuple[float, Configuration]]],
    ) -> List[Configuration]:
        # additional_start_points can be other points generated from global search

        if runhistory.empty():
            configs_previous_runs = self.config_space.sample_configuration(size=num_points)
        else:
            configs_previous_runs = runhistory.get_all_configs()

        # configurations with the highest previous EI
        configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
        configs_previous_runs_sorted = [conf[1] for conf in configs_previous_runs_sorted[:num_points]]

        if additional_start_points is not None:
            additional_start_points = [asp[1] for asp in additional_start_points[:num_points]]
        else:
            additional_start_points = []

        init_points = []
        init_points_as_set = set()  # type: Set[Configuration]
        for cand in itertools.chain(
                init_points,
                configs_previous_runs_sorted,
                additional_start_points,
        ):
            if cand not in init_points_as_set:
                init_points.append(cand)
                init_points_as_set.add(cand)

        return init_points

    def _do_search(
            self,
            start_points: List[Configuration],
    ) -> List[Tuple[float, Configuration]]:
        # for each start point, return one corresponding local_search result
        # Gather data structure for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        candidates = start_points

        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates = self.acquisition_function(candidates)
        if num_candidates == 1:
            acq_val_candidates = [acq_val_candidates[0]]
        else:
            acq_val_candidates = [a for a in acq_val_candidates]

        def replace_candidate(_candidate, _acq_val, _best_acq_val, _func):
            _new_candidate = _candidate
            _new_acq_val = _acq_val

            gen = get_one_exchange_neighbourhood(_candidate, seed=0)

            # Iterate over all neighbours and select the best one
            for _neighbour_candidate in gen:
                _acq_val_neighbour_candidate = _func([_neighbour_candidate])[0]

                if _acq_val_neighbour_candidate > _best_acq_val:
                    _new_candidate = _neighbour_candidate
                    _new_acq_val = _acq_val_neighbour_candidate
            
            return _new_candidate, _new_acq_val

        # Start of local search
        # Get neighbours of every candidate
        neighbour_candidates = []
        for i, (candidate, acq_val) in enumerate(zip(candidates.copy(), acq_val_candidates.copy())):
            
            old_candidate = candidate
            old_acq_val = best_acq_val = acq_val
            while True:
                new_candidate, new_acq_val = replace_candidate(old_candidate, old_acq_val, best_acq_val, self.acquisition_function)

                # abort if no better candidate is found
                if old_candidate == new_candidate or abs(old_acq_val - new_acq_val) < 1e-2:
                    break
                else:
                    old_candidate = new_candidate
                    old_acq_val = best_acq_val = new_acq_val

            candidates[i] = new_candidate
            acq_val_candidates[i] = new_acq_val

        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]


class GlobalAndLocalSearch(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.global_search = GlobalOptimizerSobol(acquisition_function, config_space, rng)
        self.local_search = LocalSearch(acquisition_function, config_space, rng)

    def _maximize(
        self,
        runhistory: RunHistory,
        num_points: int,
    ) -> List[Tuple[float, Configuration]]:

        global_configurations = self.global_search._maximize(runhistory, num_points)
        return self.local_search._maximize(runhistory, num_points, additional_start_points=global_configurations)
