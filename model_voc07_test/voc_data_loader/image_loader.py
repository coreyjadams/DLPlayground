from voc_meta import voc_meta
from voc_image import voc_image
import random
import numpy
# import threading


class image_loader(object):
    """docstring for image_loader"""
    def __init__(self, top_dir, rand_seed = None):
        super(image_loader, self).__init__()

        self._top_dir = top_dir

        # Read in the image meta data:
        self._meta = voc_meta(self._top_dir)

        # These are the classes to use, default (None) means all of them:
        self._classes = None

        # Set up a random number generator:
        self._random = random.Random()
        if rand_seed is not None:
            self._random.seed(rand_seed)

        self._max_roi = 10

        # Set up pointers for the current and next batch:
        self._current_labels  = None
        self._current_indexes = None
        self._current_images  = None
        self._current_bb      = None
        # self._next_labels    = None
        # self._next_images    = None

        self._image_width = 512
        self._image_height = 512


    def set_classes(self, _classes):
        self._classes = _classes
        # self._images = 


    def get_next_train_batch(self, batch_size, mode="bb"):
        # First, get the image labels need:
        _image_indexes = self._meta.train_indexes(self._classes)

        # Read in the images for this batch of data:
        _this_batch_index = self._random.sample(_image_indexes, batch_size)
 

        # Prepare output image storage:
        _out_image_arr = numpy.zeros((batch_size, self._image_height, self._image_width, 3))

        if "class" in mode:
            _out_label = numpy.zeros((batch_size, len(self._meta.classes())))
        else:
            _out_label = numpy.zeros((batch_size, self._max_roi, len(self._meta.classes())))
            _out_label[:] = 0


        _out_bb = numpy.zeros((batch_size, self._max_roi, 4))

        i = 0
        for image in _this_batch_index:
            # print image
            _xml =  "{:06d}.xml".format(image)
            img = voc_image(self._top_dir, _xml)

            # Compute the mean of the image:
            mean_rgb = numpy.mean(img.image(), axis=(0,1))
            
            # Randomly pad the image to make the output the desired dimensions:
            _delta_w = self._image_width  - img.width()
            _delta_h = self._image_height - img.height()
 
            _w_pad = self._random.randint(0, _delta_w-1)
            _h_pad = self._random.randint(0, _delta_h-1)
    
            _out_image_arr[i,:,:,:] += mean_rgb
            _out_image_arr[i, _h_pad:-(_delta_h -_h_pad),_w_pad:-(_delta_w -_w_pad), :] = img.image()

            
            j = 0
            for cat, box in zip(img.categories(), img.bounding_boxes()):
                if "class" in mode:
                    _out_label[i,self._meta.class_index(cat)] = 1
                else:
                    _out_label[i,j, self._meta.class_index(cat)] = 1
                _out_bb[i, j, 0] = box[0] + _w_pad
                _out_bb[i, j, 1] = box[1] + _h_pad
                _out_bb[i, j, 2] = box[2] + _w_pad
                _out_bb[i, j, 3] = box[3] + _h_pad
                j += 1
                if j > 9:
                    break
            
                # Set the label, too:
                # 
            i += 1

        self._current_images  = _out_image_arr
        self._current_labels  = _out_label
        self._current_bb      = _out_bb
        self._current_indexes = _this_batch_index

        return _out_image_arr, _out_label, _out_bb

    def get_next_val_batch(self, batch_size, mode="bb"):
        # First, get the image labels need:
        _image_indexes = self._meta.val_indexes(self._classes)

        # Read in the images for this batch of data:
        _this_batch_index = self._random.sample(_image_indexes, batch_size)

        # Prepare output image storage:
        _out_image_arr = numpy.zeros((batch_size, self._image_height, self._image_width, 3))

        if "class" in mode:
            _out_label = numpy.zeros((batch_size, len(self._meta.classes())))
        else:
            _out_label = numpy.zeros((batch_size, self._max_roi, len(self._meta.classes())))
            _out_label[:] = 0


        _out_bb = numpy.zeros((batch_size, self._max_roi, 4))

        i = 0
        for image in _this_batch_index:
            # print image
            _xml =  "{:06d}.xml".format(image)
            img = voc_image(self._top_dir, _xml)

            # Randomly pad the image to make the output the desired dimensions:
            _delta_w = self._image_width  - img.width()
            _delta_h = self._image_height - img.height()
 
            _w_pad = self._random.randint(0, _delta_w-1)
            _h_pad = self._random.randint(0, _delta_h-1)

            _out_image_arr[i, _h_pad:-(_delta_h -_h_pad),_w_pad:-(_delta_w -_w_pad), :] = img.image()

            
            j = 0
            for cat, box in zip(img.categories(), img.bounding_boxes()):
                if "class" in mode:
                    _out_label[i,self._meta.class_index(cat)] = 1
                else:
                    _out_label[i,j, self._meta.class_index(cat)] = 1
                _out_bb[i, j, 0] = box[0] + _w_pad
                _out_bb[i, j, 1] = box[1] + _h_pad
                _out_bb[i, j, 2] = box[2] + _w_pad
                _out_bb[i, j, 3] = box[3] + _h_pad
                j += 1
                if j > 9:
                    break
            
                # Set the label, too:
                # 
            i += 1

        self._current_images  = _out_image_arr
        self._current_labels  = _out_label
        self._current_bb      = _out_bb
        self._current_indexes = _this_batch_index

        return _out_image_arr, _out_label, _out_bb
    
    
    def get_next_train_image(self, mode="bb", pad = False):
        # First, get the image labels need:
        _image_indexes = self._meta.train_indexes(self._classes)

        # Read in the images for this batch of data:
        image,  = self._random.sample(_image_indexes, 1)
        print "IMAGE:" +str(image)

        # print image
        _xml =  "{:06d}.xml".format(image)
        img = voc_image(self._top_dir, _xml)

        n_images = len(img.bounding_boxes()) 

        if pad:
            n_images = self._max_roi

        # Prepare output image storage:
        _out_image_arr = numpy.zeros((self._image_height, self._image_width, 3))

        _out_label = numpy.zeros((n_images, len(self._meta.classes())))
        _out_label[:] = 0

        _out_bb = numpy.zeros((n_images, 4))

        # Randomly pad the image to make the output the desired dimensions:
        _delta_w = self._image_width  - img.width()
        _delta_h = self._image_height - img.height()

        _w_pad = self._random.randint(0, _delta_w-1)
        _h_pad = self._random.randint(0, _delta_h-1)

        _out_image_arr[_h_pad:-(_delta_h -_h_pad),_w_pad:-(_delta_w -_w_pad), :] = img.image()

        
        j = 0
        for cat, box in zip(img.categories(), img.bounding_boxes()):
            _out_label[j, self._meta.class_index(cat)] = 1
            _out_bb[j, 0] = box[0] + _w_pad
            _out_bb[j, 1] = box[1] + _h_pad
            _out_bb[j, 2] = box[2] + _w_pad
            _out_bb[j, 3] = box[3] + _h_pad
            j += 1
        
            # Set the label, too:
            # 

        self._current_images  = _out_image_arr
        self._current_labels  = _out_label
        self._current_bb      = _out_bb
        self._current_indexes = [image]

        return _out_image_arr, _out_label, _out_bb