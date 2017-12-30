import numpy as np


class Perceptron:
    """
    Numpy implementation of the single-layer Perceptron neural network.
    """

    hyperparameters = {'AND': {'weights': [1, 1], 'theta': 1.5},
                       'NOT': {'weights': [-1], 'theta': -0.5},
                       'OR': {'weights': [1, 1], 'theta': 0.5},
                       'ARBITRARY': {'weights': [2, -2, 2], 'theta': 1}
                       }

    def __init__(self, function):
        self._function = function.upper()
        if self._function not in self.hyperparameters:
            raise ValueError('Function "{}" is not in {}'.format(
                self._function, str(self.hyperparameters.keys())[10:-1])
            )

    @property
    def weights(self):
        return self.hyperparameters[self._function]['weights']

    @property
    def theta(self):
        return self.hyperparameters[self._function]['theta']

    def output(self, x):
        return np.heaviside(np.dot(x, self.weights) - self.theta, 0)
