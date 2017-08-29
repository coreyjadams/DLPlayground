# import tensorflow as tf
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
        self._key_param_dict = dict()

        self._key_param_dict.update({'base_lr' : 'blr' ,
                                     'lr_decay' : 'lrd' ,
                                     'decay_step' : 'ds' })


    def training_params(self):
        return self._training_params

    def network_params(self):
        return self._network_params


    def get_string(self):
        s = ""
        for key in self._key_param_dict:
            if key in self._training_params:
                s += "_{}_{}".format(self._key_param_dict[key], self._training_params[key])
            if key in self._network_params:
                s += "_{}_{}".format(self._key_param_dict[key], self._network_params[key])
        return s

class network(object):
    """docstring for network"""

    def __init__(self, name, hyperparams):
        super(network, self).__init__()
        self._name = name
        self._params = hyperparams

    def name(self):
        return self._name

    def full_name(self):
        _full_name = self._name + self._params.get_string()
        return _full_name

    def hyperparameters(self):
        return self._params

    def build_network(self, input_tensor):
        output = input_tensor
        return output
