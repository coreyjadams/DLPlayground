
class training_parameters(object):

    def __init__(self):
        self._training_params = dict()
        self._key_param_dict = dict()
        self._key_param_dict.update({'base_lr' : 'blr' ,
                                     'lr_decay' : 'lrd' ,
                                     'decay_step' : 'ds' })

    def training_params(self):
        return self._training_params


    def get_string(self):
        s = ""
        for key in self._key_param_dict:
            if key in self._training_params:
                s += "_{}_{}".format(self._key_param_dict[key], self._training_params[key])
        return s
        