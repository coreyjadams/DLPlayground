import tensorflow as tf
import numpy
import sys
from models import rpn, rpn_utils, resnet, training_params

from voc_data_loader import voc_meta, image_loader

# Use a resnet for the convolutional step:
conv_params = resnet.resnet_params()
conv_params.network_params()['n_blocks'] = 4
conv_params.network_params()['include_final_classifier'] = False
conv_params.network_params()['n_classes'] = 10
conv_params.network_params()['n_initial_filters'] = 12
conv_params.network_params()['downsample_interval'] = 1
conv_params.network_params()['initial_stride'] = 2
conv_params.network_params()['initial_kernel'] = 5
conv_params.network_params()['bottleneck'] = False
conv_params.network_params()['weight_decay'] = 1E-3
conv_params.network_params()['activation'] = 'softmax'


# Set up the network we want to test:
rpn_params = rpn.rpn_params()
rpn_params.network_params()['n_anchors_per_box'] = 9
rpn_params.network_params()['weight_decay'] = 1E-3
rpn_params.network_params()['n_selected_regressors'] = 128

train_params = dict()
train_params['LOGDIR'] = "logs/rpn_resnet/"
train_params['ITERATIONS'] = 5000
train_params['SAVE_ITERATION'] = 100
train_params['RESTORE'] = False
train_params['RESTORE_INDEX'] = -1


# Set up the graph:
with tf.Graph().as_default():


    N_MAX_TRUTH = 10
    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [1, 512,512,3], name='x')
    label_tensor = tf.placeholder(tf.float32, [N_MAX_TRUTH, 20], name='labels')
    box_label = tf.placeholder(tf.float32, [N_MAX_TRUTH, 4], name='truth_anchors')





    # Let the convolutional part of the network be independant
    # of the classifiers:
    
    conv_net = resnet.resnet(conv_params)
    final_conv_layer = conv_net.build_network(input_tensor=data_tensor,
                                              is_training=True)    

    RPN = rpn.rpn(rpn_params)
    classifier, regressor = RPN.build_rpn(final_conv_layer=final_conv_layer, 
                                              is_training=True)

    # The previous functions work with batch_size == 1 to allow interface with
    # other code, particularly for the conv nets.
    # 
    # Here, squeeze the batch size out to get just the raw shapes:
    classifier = tf.squeeze(classifier)
    regressor = tf.squeeze(regressor)



    # Get a set of reference anchors:

    n_anchors_x = final_conv_layer.get_shape().as_list()[1]
    n_anchors_y = final_conv_layer.get_shape().as_list()[2]
    effective_stride_x = data_tensor.get_shape().as_list()[1] / n_anchors_x
    effective_stride_y = data_tensor.get_shape().as_list()[2] / n_anchors_y
    n_anchors_x -= 2
    n_anchors_y -= 2


    _base_anchors = rpn_utils.generate_anchors(base_size = 16*3, 
                                               ratios = [0.5, 1, 2.0], 
                                               scales = [2,4,6])
    # _base_anchors = rpn_utils.generate_anchors(base_size = 16*3, 
    #                                            ratios = [1], 
    #                                            scales = [4, 8])    

    _anchors = rpn_utils.pad_anchors(_base_anchors,
                                    n_tiles_x=n_anchors_x, 
                                    n_tiles_y=n_anchors_y, 
                                    step_size_x=effective_stride_x, 
                                    step_size_y=effective_stride_y)

    # For debugging only: force the output of the regressor to a deterministic value:
    regressor = tf.placeholder(tf.float32, (8100, 4), name = "fake_regressor")
    numpy.random.seed(0)
    _fake_regressor = numpy.random.rand(8100, 4)
    _fake_regressor[:,2] *= 0.25
    _fake_regressor[:,3] *= 0.25

    rpn_utils.boxes_whctrs_to_minmax(_anchors, in_place=True)

    # Anchors are now in min/max format and will always be 
    # assumed to be in min/max format

    anchors = tf.placeholder(tf.float32, _anchors.shape, name="anchors")

    # Downselect and find the positive and negative indexes
    # based on IoU and non max suppression:
    pos_box_ind, pos_true_ind, neg_box_ind, neg_true_ind = RPN.downselect(
        regressor, classifier, anchors, box_label)

    # #Above, pos_true and neg_true is the matched truth box index for the 
    # #positive and negative examples
    
    regression_loss, other = RPN.regression_loss(regressor,
        pos_box_ind, pos_true_ind, box_label, anchors)

    # Last step is to compute the loss for regression and classification
    # For the regression loss, slice off only the positive examples
    # and the

    LOGDIR = train_params['LOGDIR'] + "/" + RPN.full_name() + "/"


    # Add a global step accounting for saving and restoring training:
    with tf.name_scope("global_step") as scope:
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    # # Add cross entropy (loss)
    # with tf.name_scope("cross_entropy") as scope:
    #     cross_entropy = tf.reduce_mean(
    #         tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,
    #                                                 logits=logits))
    #     loss_summary = tf.summary.scalar("Loss", cross_entropy)

    # # Add accuracy:
    # with tf.name_scope("accuracy") as scope:
    #     correct_prediction = tf.equal(
    #         tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     acc_summary = tf.summary.scalar("Accuracy", accuracy)
    #
    # # Set up a learning rate so we can adapt it:
    # learning_rate = tf.train.exponential_decay(learning_rate=params.training_params()['base_lr'], 
    #                                            global_step=global_step,
    #                                            decay_steps=params.training_params()['decay_step'],
    #                                            decay_rate=params.training_params()['lr_decay'],
    #                                            staircase=True)
    
    # lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    # # Set up a training algorithm:
    # with tf.name_scope("training") as scope:
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         train_step = tf.train.AdamOptimizer(learning_rate).minimize(
    #             cross_entropy, global_step=global_step)




    merged_summary = tf.summary.merge_all()

    # # Set up a saver:
    train_writer = tf.summary.FileWriter(LOGDIR)


    tf.set_random_seed(0)
    print "Initialize session ..."
    with tf.Session() as sess:
        
        # train_writer.add_graph(sess.graph)

        # if not train_params.training_params()['RESTORE']:
        sess.run(tf.global_variables_initializer())
        #     train_writer.add_graph(sess.graph)
        #     saver = tf.train.Saver()
        # else: 
        #     latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
        #     print latest_checkpoint
        #     saver = tf.train.Saver()
        #     saver.restore(sess, latest_checkpoint)


        print "Begin training ..."
        # Run training loop
        # while not sv.should_stop():
        step = 0
        loader = image_loader("./", rand_seed=1)

        # while step < params.training_params()['ITERATIONS']:
        for i in xrange(5):
            # step = sess.run(global_step)

            data, labels, boxes = loader.get_next_train_batch(1)
            labels = numpy.squeeze(labels)
            boxes = numpy.squeeze(boxes)
            matched_truths, matched_anchors = sess.run(
                [regression_loss, other], 
                feed_dict={data_tensor : data,
                           label_tensor : labels,
                           box_label : boxes,
                           regressor : _fake_regressor, 
                           anchors : _anchors})

            print matched_truths.shape
            print matched_anchors.shape
            # print "pos_reg_coords.shape: {}".format(pos_reg_coords.shape)
            # print "pos_scores_raw_count: {}".format(pos_scores_raw_count)
            # print "pos_scores_update.shape: {}".format(pos_scores_update.shape)
            # print "ious_above_threshold: {}".format(ious_above_threshold)
            # print ""
            # print reg_loss

            # print training accuracy every 10 steps:
            # if i % 10 == 0:
            #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary],
            #                                                       feed_dict={data_tensor:data,
            #                                                                  label_tensor:label})
            #     train_writer.add_summary(loss_s,i)
            #     train_writer.add_summary(accuracy_s,i)

                # sys.stdout.write('Training in progress @ step %d accuracy %g\n' % (i,training_accuracy))
                # sys.stdout.flush()

            # if step != 0 and step % 5 == 0:
            #     print "Running Summary"
            #     _, summ = sess.run([merged_summary], feed_dict={data_tensor: data, 
            #                                      label_tensor: label})
            #     print "Saving Summary"
            #     sv.summary_computed(sess, summ)
            # [loss, acc, _, summary] = sess.run([cross_entropy, accuracy, train_step, merged_summary], 
            #                           feed_dict={data_tensor: data, 
            #                                      label_tensor: label})
            # train_writer.add_summary(summary, step)



            # # train_writer.add_summary(summary, i)
            # sys.stdout.write(
            #     'Training in progress @ step %d, loss %g, accuracy %g\r' % (step, loss, acc))
            # # sys.stdout.flush()

        # print "\nFinal training loss {}, accuracy {}".format(loss, acc)
        # data, label = mnist.test.next_batch(2000)
        
        # [l, a, summary] = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
        #         data_tensor: data, label_tensor: label})
        # print "\nTesting loss {}, accuracy {}".format(l, a)
