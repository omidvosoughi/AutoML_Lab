from ConfigSpace import Configuration
import typing

from scipy.stats import norm
import numpy as np


from util import convert_configurations_to_array, impute_nan

class AbstractAcquisitionFunction(object):
    def __init__(self, model):
        self.model = model
        self._required_updates = ('model', )

    def update(self, **kwargs: typing.Any) -> None:
        for key in self._required_updates:
            if key not in kwargs:
                raise ValueError(
                    'Acquisition function %s needs to be updated with key %s, but only got '
                    'keys %s.'
                    % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def __call__(self, configurations: typing.List[Configuration]) -> np.ndarray:
        X = convert_configurations_to_array(configurations)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    def _compute(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()



class EI(AbstractAcquisitionFunction):
    def __init__(self,
                 model,
                 par: float = 0.0):
        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None # best current observation
        self._required_updates = ('model', 'eta')

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 1e-9
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        return f
