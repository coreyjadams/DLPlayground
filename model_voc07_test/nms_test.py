import tensorflow as tf
import sys

import numpy
from voc_data_loader import voc_meta, image_loader

from models import rpn_utils

# Set up the graph:
with tf.Graph().as_default():

    _anchors = rpn_utils.pad_anchors(rpn_utils.generate_anchors(scales = numpy.asarray([4,8,16])))
    anchors = tf.placeholder(tf.float32, _anchors.shape, name="anchors")

    n_anchors = _anchors.shape[0]

    scores = tf.random_uniform((n_anchors,), minval=0, maxval=1.0)

    # Fake gather the anchors based on score:
    pos_anchor_indexes = tf.where(scores > 1.1)

    pos_anchors = tf.squeeze(tf.gather(anchors, [pos_anchor_indexes]))
    pos_scores = tf.squeeze(tf.gather(scores, [pos_anchor_indexes]))

    selected = tf.image.non_max_suppression(pos_anchors, pos_scores, 
                                            max_output_size=100, 
                                            iou_threshold=0.7)

    # No need to actually remove the empty elements, just need to operate on them
    # Currently anchors are set as (x_center, y_center, width, height):
    




    with tf.Session() as sess:

        pos_anchor, pos_scores, selected_anchors = sess.run([pos_anchors, pos_scores, selected], feed_dict={anchors : _anchors})
        
        print pos_anchor.shape
        print selected_anchors.shape