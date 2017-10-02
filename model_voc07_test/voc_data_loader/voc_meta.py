import pandas as pd
import numpy

# This class reads in the meta data for the VOC07 dataset and let's you slice and dice it.
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
        
    def classes(self):
        return self._classes

    def class_index(self, _class):
        if _class in self._classes:
            return self._classes.index(_class)
