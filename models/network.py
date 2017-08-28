import tensorflow as tf

"""
This class is meant to make things easier for me to build networks
and test hyperparameters.  Each "real" network will inherit from this 
base structure, which also has a few utility layers (like conv2d, etc)
"""


class hyperparameters(object):
    """docstring for hyperparameters"""

    def __init__(self):
        super(hyperparameters, self).__init__()
        self._network_params = dict()
        self._training_params = dict()
        self._key_param_list = []

        # Some basic initialization stuff:
        self._params_dict['learning_rate' : 1E-3]

    def network_params(self):
        return self._params_dict

    def add_param(self, param, value):
        self._params_dict[param] = value
        return

    def get_string(self):
        s = ""
        for key in self._key_param_list:
            if key in self._training_params:
                s += "{}_{}_".format(key, self._training_params[key])
            if key in self._network_params:
                s += "{}_{}_".format(key, self._network_params[key])


class network(object):
    """docstring for network"""

    def __init__(self, name, hyperparams):
        super(network, self).__init__()
        self._name = name
        self._hyperparameters = hyperparams

    def name(self):
        return self._name

    def full_name(self):
        _full_name = self._name + "_" + self._hyperparameters.get_string()
        return _full_name

    def hyperparameters(self):
        return self._hyperparameters

    def build_network(self, input_tensor):
        output = input_tensor
        return output

