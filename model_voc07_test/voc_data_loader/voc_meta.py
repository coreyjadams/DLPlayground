import pandas as pd
import numpy

# This class reads in the meta data for the VOC dataset and let's you slice and dice it.
# It doesn't read in any images, just the meta once at initialization
# and then you can load individual images in another scipt.

class voc_meta(object):
    """docstring for voc_meta"""
    def __init__(self, top_dir):
        super(voc_meta, self).__init__()
        
        # 
        self._classes= ['aeroplane', 
                        'bicycle',
                        'bird',
                        'boat',
                        'bottle',
                        'bus',
                        'car',
                        'cat',
                        'chair',
                        'cow',
                        'diningtable',
                        'dog',
                        'horse',
                        'motorbike',
                        'person',
                        'pottedplant',
                        'sheep',
                        'sofa',
                        'train',
                        'tvmonitor']
        self._top_dir = top_dir
        _f_base = self._top_dir + '/ImageSets/Main/'
        # Read in the train, val, and trainval total sets:
        self._df_train    = pd.read_csv(_f_base + 'train.txt', 
                                        delim_whitespace = True,
                                        index_col=0,
                                        names={'TRAIN'}).replace([-1, 1], [False, True])
        self._df_val      = pd.read_csv(_f_base + 'val.txt', 
                                        delim_whitespace = True,
                                        index_col=0,
                                        names={'VAL'}).replace([-1, 1], [False, True])
        self._df_trainval = pd.read_csv(_f_base + 'trainval.txt', 
                                        delim_whitespace = True,
                                        index_col=0,
                                        names={'TRAINVAL'}).replace([-1, 1], [False, True])

        for _class in self._classes:
            self.read_class(_class)

        # Read in the list of segmenatation images:
        _f_base = self._top_dir + '/ImageSets/Segmentation/'
        self._seg_train    = numpy.loadtxt(_f_base + "train.txt", dtype=int)
        self._seg_val      = numpy.loadtxt(_f_base + "val.txt", dtype=int)
        self._seg_trainval = numpy.loadtxt(_f_base + "trainval.txt", dtype=int)


    def classes(self):
        return self._classes

    def class_index(self, _class):
        if _class in self._classes:
            return self._classes.index(_class)

    def class_name(self, index):
        return self._classes[index]
            
    def read_class(self, _class):
        _f_base = self._top_dir + '/ImageSets/Main/{}_'.format(_class)
        _df_train    = pd.read_csv(_f_base + 'train.txt', 
                                    delim_whitespace = True,
                                    index_col=0,
                                    names={_class}).replace([-1, 1], [False, True])
        _df_val      = pd.read_csv(_f_base + 'val.txt', 
                                    delim_whitespace = True,
                                    index_col=0,
                                    names={_class}).replace([-1, 1], [False, True])
        _df_trainval = pd.read_csv(_f_base + 'trainval.txt', 
                                    delim_whitespace = True,
                                    index_col=0,
                                    names={_class}).replace([-1, 1], [False, True])
        
        self._df_train    = pd.merge(self._df_train, _df_train, 
                                    left_index=True, 
                                    right_index=True, 
                                    how='outer').fillna(False)
        self._df_val      = pd.merge(self._df_val, _df_val, 
                                    left_index=True, 
                                    right_index=True, 
                                    how='outer').fillna(False)
        self._df_trainval = pd.merge(self._df_trainval, _df_trainval, 
                                    left_index=True, 
                                    right_index=True, 
                                    how='outer').fillna(False)
    
    def train_indexes(self, _select_classes = None, _seg_indexes=False):
        if _select_classes is None:
            if not _seg_indexes:
                return self._df_train.index
            else:
                return self._seg_train
        else:
            _query = ""
            for _class in _select_classes:
                _query += "{} == True &".format(_class)
            _query = _query.rstrip("&")
            if not _seg_indexes:
                return self._df_train.query(_query).index
            else:
                return numpy.intersect1d(self._df_train.query(_query).index, 
                                         self._seg_train)

    def val_indexes(self, _select_classes = None, _seg_indexes=False):
        if _select_classes is None:
            if not _seg_indexes:
                return self._df_val.index
            else:
                return self._seg_val
        else:
            _query = ""
            for _class in _select_classes:
                _query += "{} == True &".format(_class)
            _query = _query.rstrip("&")
            if not _seg_indexes:
                return self._df_val.query(_query).index
            else:
                return numpy.intersect1d(self._df_val.query(_query).index, 
                                         self._seg_val)

    def indexes(self, _select_classes = None, _seg_indexes=False):
        if _select_classes is None:
            if not _seg_indexes:
                return self._df_trainval.index
            else:
                return self._seg_trainval
        else:
            _query = ""
            for _class in _select_classes:
                _query += "{} == True &".format(_class)
            _query = _query.rstrip("&")
            if not _seg_indexes:
                return self._df_trainval.query(_query).index
            else:
                return numpy.intersect1d(self._df_trainval.query(_query).index, 
                                         self._seg_trainval)
