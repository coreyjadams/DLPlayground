import numpy
from bs4 import BeautifulSoup
from scipy import ndimage
import os

class voc_image(object):

    def __init__(self, top_dir, xml_file):
        super(voc_image, self).__init__()


        # Read the xml file:
        with open(top_dir + "/Annotations/" + xml_file) as _raw_xml:
            self._xml = BeautifulSoup(_raw_xml.read(), 'xml')

        self._width = int(self._xml.find("width").get_text())
        self._height = int(self._xml.find("height").get_text())
        self._depth = int(self._xml.find("depth").get_text())

        self._categories = []
        self._bb = []
        self._pose = []
        self._difficult = []
        self._truncated = []
        # Determine the main classes:
        for obj in self._xml.findAll("object"):
            self._categories.append(obj.findChild("name").get_text())
            box = obj.findChild("bndbox")
            self._bb.append([])
            self._bb[-1].append(int(box.findChild('xmin').get_text()))
            self._bb[-1].append(int(box.findChild('ymin').get_text()))
            self._bb[-1].append(int(box.findChild('xmax').get_text()))
            self._bb[-1].append(int(box.findChild('ymax').get_text()))
            self._pose.append(obj.findChild('truncated').get_text())
            self._difficult.append(bool(int(obj.findChild('difficult').get_text())))
            self._truncated.append(bool(int(obj.findChild('truncated').get_text())))

        self._has_segmentation = bool(int(self._xml.find("segmented").get_text()))

        # Load the images:
        _base_imname = xml_file.replace(".xml", "") + ".jpg"
        self._base_image = ndimage.imread(top_dir + "/JPEGImages/" + _base_imname, mode='RGB') * (1./255)
        if self._has_segmentation:
            _base_imname = xml_file.replace(".xml", "") + ".png"
            self._seg_class = ndimage.imread(top_dir + "/SegmentationClass/" + _base_imname,mode='P')
            self._seg_obj = ndimage.imread(top_dir + "/SegmentationObject/" + _base_imname,mode='P')


    def image(self):
        return self._base_image

    def segmentation_object(self):
        return self._seg_obj

    def segmentation_class(self):
        return self._seg_class

    def xml(self):
        return self._xml

    def categories(self):
        return self._categories

    def has_segmentation(self):
        return self._has_segmentation

    def n_boxes(self):
        return len(self._bb)

    def bounding_boxes(self):
        return self._bb

    def pose(self, index):
        return self._pose[index]

    def truncated(self, index):
        return self._truncated[index]

    def difficult(self, index):
        return self._difficult[index]

    def width(self):
        return self._width

    def height(self):
        return self._height
    
    def depth(self):
        return self._depth
