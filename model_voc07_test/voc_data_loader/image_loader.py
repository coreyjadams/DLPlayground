


class voc_base(object):
    """docstring for voc_base"""
    def __init__(self):
        super(voc_base, self).__init__()
        
        self._root_dir = os.environ['DLPLAYGROUND'] + "/model_voc07_test/VOC2007/"
        self._img_dir = os.path.join(root_dir, 'JPEGImages/')
        self._ann_dir = os.path.join(root_dir, 'Annotations')
        self._set_dir = os.path.join(root_dir, 'ImageSets')




class image_loader(voc_base):
    """docstring for image_loader"""
    def __init__(self):
        super(image_loader, self).__init__()
        self._train_images = []
        self._val_images = []
        self._train_labels = []
        self._val_labels = []
        self._load_images()

    def _load_images(self):
        for _class in list_image_sets():
            self._train_images.append()
            print imgs_from_category(_class, "train")

