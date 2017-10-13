import tensorflow as tf
import sys

import numpy
from voc_data_loader import voc_meta, image_loader

from models import rpn

# Set up the graph:
with tf.Graph().as_default():

    MAX_N_TRUTH_BOXES = 10

    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [512,512,3], name='x')
    label_tensor = tf.placeholder(tf.float32, [MAX_N_TRUTH_BOXES,  20], name='labels')
    box_label = tf.placeholder(tf.float32, [MAX_N_TRUTH_BOXES, 4], name='truth_anchors')

    # positive_labels = tf.reduce_sum(label_tensor, axis=-1)

    _anchors = rpn.pad_anchors(rpn.generate_anchors(scales = numpy.asarray([4,8,16])))
    anchors = tf.placeholder(tf.float32, _anchors.shape, name="anchors")


    # # mask = tf.equal(positive_labels, empty)

    # true_label_indexes = tf.where(positive_labels > 0)

    # forced_mask = [[True, True, False, False, False, False, False, False, False,False],
    #                # [True, True, True, False, False, False, False, False, False,False],
    #                # [True, True, False, False, False, False, False, False, False,False],
    #                # [True, True, False, False, False, False, False, False, False,False]
    #                ]
    ious = rpn.compute_IoU(box_label, anchors)

    # No need to actually remove the empty elements, just need to operate on them
    # Currently anchors are set as (x_center, y_center, width, height):
    


    loader = image_loader("./", rand_seed=0)

    with tf.Session() as sess:

        data, labels, boxes = loader.get_next_train_image()

        # print labels
        print "True box 1 {}".format(boxes[0])
        print "True box 2 {}".format(boxes[1])
        print "True box 3 {}".format(boxes[2])

        print numpy.unique(_anchors[:2])

        ious = sess.run(ious, feed_dict={label_tensor : labels,
                                         box_label : boxes, 
                                         anchors : _anchors})
        print  "Computed ious shape: {}".format(ious.shape)

        print "Index of iou over 0.7 for true box 1: {}".format(numpy.where(ious[0,:] > 0.7))
        print "Index of iou over 0.7 for true box 2: {}".format(numpy.where(ious[1,:] > 0.7))
        # print _anchors[6019]

        # max_index = numpy.unravel_index(numpy.argmax(ious), ious.shape)
        # print rpn.single_IoU(boxes[1], _anchors[3])

        # print ious[max_index]

        # Possible method: pull IoUs from tf, figure out which indexes get positive and
        # which get negative values (easier in numpy, with conditionals), then mask 
        # tensorflow tensors for computation of loss (reg and cls)
        # Mask can be an array of 0.0 or 1.0 with same shape as the reg and cls output.
        # reg mask is only 1.0 for the positive examples (IoU > 0.7 OR max IoU if all 
        # below threshold)
        # cls mask is 1.0 for positive OR negative examples, and 0.0 otherwise.
        # 
        # Still need to do nonmaximal supression


        # print boxes

        # indexes = sess.run([positive_labels], feed_dict={label_tensor : labels,box_label : boxes})

        # print len(boxes)
        # print len(_anchors)
        # for i in xrange(len(boxes)):
        #     for j in xrange(len(_anchors)):
        #         print "Box {}, Anchor {}, index {}: {} - {} = {}".format(
        #             i, j, i*9 + j,
        #             rpn.single_IoU(boxes[i], _anchors[j]),
        #             ious[i][j],
        #             rpn.single_IoU(boxes[i], _anchors[j]) - ious[i][j])
        # print ious
