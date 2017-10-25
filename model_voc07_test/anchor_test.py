# import tensorflow as tf
import sys

import numpy
from voc_data_loader import voc_meta, image_loader

from models import rpn_utils


MAX_N_TRUTH_BOXES = 10

# positive_labels = tf.reduce_sum(label_tensor, axis=-1)

_anchors = rpn_utils.pad_anchors(
                rpn_utils.generate_anchors(base_size = 16*3, 
                                           ratios = [0.5, 1, 2.0], 
                                           scales = [2,4,8]))

_anchors = rpn_utils.boxes_whctrs_to_minmax(_anchors)

_anchors = rpn_utils.prune_noninternal_anchors(_anchors, [0,0,512,512])


seed = int(5000*numpy.random.rand(1))
print "seed is {}".format(seed)
loader = image_loader("VOC2007",seed)
# loader = image_loader("VOC2007",3300)
# loader = image_loader("VOC2007",2227)
data, labels, boxes = loader.get_next_train_image()
# labels = 

boxes = numpy.squeeze(boxes)


labels, ground_truths, matched_anchors = rpn_utils.numpy_select_label_anchors_minmax(boxes, _anchors)



print "Positive labels: {}".format(numpy.count_nonzero(labels == 1))
print "Negative labels: {}".format(numpy.count_nonzero(labels == 0))
print "Don't Care labels: {}".format(numpy.count_nonzero(labels == -1))
print "Labels shape: {}".format(labels.shape)