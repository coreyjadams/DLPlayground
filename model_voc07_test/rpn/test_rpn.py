import tensorflow as tf
import numpy
import sys
from models import rpn, rpn_utils, resnet, training_params

from voc_data_loader import voc_meta, image_loader

# Use a resnet for the convolutional step:
conv_params = resnet.resnet_params()
conv_params.network_params()['n_blocks'] = 8
conv_params.network_params()['include_final_classifier'] = False
conv_params.network_params()['n_classes'] = 10
conv_params.network_params()['n_initial_filters'] = 12
conv_params.network_params()['downsample_interval'] = 2
conv_params.network_params()['initial_stride'] = 2
conv_params.network_params()['initial_kernel'] = 5
conv_params.network_params()['bottleneck'] = False
conv_params.network_params()['weight_decay'] = 1E-3
conv_params.network_params()['activation'] = 'softmax'
conv_params.network_params()['restore_file'] = "/home/cadams/DLPlayground/model_voc07_test/logs/rpn_resnet/resnet_ik_5_nb_8_di_2_is_2_B_False_nf_12//checkpoints/save-12000"

# Set up the network we want to test:
rpn_params = rpn.rpn_params()
rpn_params.network_params()['n_anchors_per_box'] = 9
rpn_params.network_params()['weight_decay'] = 1E-3
rpn_params.network_params()['n_selected_regressors'] = 56

train_params = dict()
train_params['LOGDIR'] = "logs/rpn_resnet/"
train_params['ITERATIONS'] = 100000
train_params['SAVE_ITERATION'] = 5000
train_params['RESTORE'] = False
train_params['RESTORE_INDEX'] = -1
train_params['LEARNING_RATE'] = 0.0001

# # Set up the graph:
# with tf.Graph().as_default():

#     N_MAX_TRUTH = 10
#     # Set input data and label for training
#     data_tensor = tf.placeholder(tf.float32, [1, 512,512,3], name='x')
#     label_tensor = tf.placeholder(tf.float32, [N_MAX_TRUTH, 20], name='labels')
#     box_label = tf.placeholder(tf.float32, [N_MAX_TRUTH, 4], name='truth_anchors')

#     # Let the convolutional part of the network be independant
#     # of the classifiers:
    
#     final_conv_layer = conv_net.build_network(input_tensor=data_tensor,
#                                               is_training=True)    
#     conv_names = tf.trainable_variables()
    

_anchors = rpn_utils.pad_anchors(
                rpn_utils.generate_anchors(base_size = 16*3, 
                                           ratios = [0.5, 1, 2.0], 
                                           scales = [2,4,6]))

_anchors = rpn_utils.boxes_whctrs_to_minmax(_anchors)

_external_anchor_indexes =  np.where(
        (_anchors[:, 0] >= 0) &
        (_anchors[:, 1] >= 0) &
        (_anchors[:, 2] <  512) &  # width
        (_anchors[:, 3] <  512)    # height
    )[0]

with tf.Graph().as_default():
    
    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [1, 512,512,3], name='x')
    # label_tensor = tf.placeholder(tf.float32, [N_MAX_TRUTH, 20], name='labels')
    # box_label = tf.placeholder(tf.float32, [N_MAX_TRUTH, 4], name='truth_anchors')
    
    # Need a tensorflow placeholder for the used anchors,
    # for the matching ground truths, and for the pos/neg labels
    
    gt_tensor = tf.placeholder(tf.float32, [rpn_params.network_params()['n_selected_regressors'], 4], name='ground_truths')
    anchor_tensor = tf.placeholder(tf.float32, [rpn_params.network_params()['n_selected_regressors'], 4], name='anchors')
    label_tensor = tf.placeholder(tf.float32, [rpn_params.network_params()['n_selected_regressors']], name='labels')
    
    conv_net = resnet.resnet(conv_params)
    with tf.variable_scope("ResNet"):
        final_conv_layer = conv_net.build_network(input_tensor=data_tensor,
                                                  is_training=False)  
    
    conv_names = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ResNet")

    
    RPN = rpn.rpn(rpn_params)
    classifier, regressor = RPN.build_rpn(final_conv_layer=final_conv_layer, 
                                              is_training=True)


    
    # The previous functions work with batch_size == 1 to allow interface with
    # other code, particularly for the conv nets.
    # 
    # Here, squeeze the batch size out to get just the raw shapes:
    classifier = tf.squeeze(classifier)
    regressor = tf.squeeze(regressor)

    n_anchors_total = len(_anchors)

    classifier = tf.reshape(classifier, (n_anchors_total, 2))
    regressor = tf.reshape(regressor, (n_anchors_total, 4))
    
    

    # Anchors are now in min/max format and will always be 
    # assumed to be in min/max format

    anchors = tf.placeholder(tf.float32, _anchors.shape, name="anchors")

    
        
    regression_loss = RPN.regression_loss(regressor,
        label_tensor, anchors, ground_truths)

    regression_loss = (1./10) * regression_loss
    tf.summary.scalar("Regression Loss", regression_loss)


    # Last step is to compute the loss for classification

    classification_loss = RPN.classification_loss(classifier, label_tensor)

    tf.summary.scalar("Classification Loss", classification_loss)

    LOGDIR = train_params['LOGDIR'] + "/" + RPN.full_name() + "/"


    # Add a global step accounting for saving and restoring training:
    with tf.name_scope("global_step") as scope:
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    # Create the global loss value:
    with tf.name_scope("total_loss"):
        total_loss = classification_loss + (1./10)*regression_loss
        tf.summary.scalar("Total Loss", total_loss)
    # # Add accuracy:
    # with tf.name_scope("accuracy") as scope:
    #     correct_prediction = tf.equal(
    #         tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     acc_summary = tf.summary.scalar("Accuracy", accuracy)
    #
    # Set up a learning rate so we can adapt it:
    # learning_rate = tf.train.exponential_decay(learning_rate=params.training_params()['base_lr'], 
    #                                            global_step=global_step,
    #                                            decay_steps=params.training_params()['decay_step'],
    #                                            decay_rate=params.training_params()['lr_decay'],
    #                                            staircase=True)
    # lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
    # Set up a training algorithm:
    with tf.name_scope("training") as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(train_params['LEARNING_RATE'], beta1=0.999)
            train_step = optimizer.minimize(total_loss, global_step=global_step)




    merged_summary = tf.summary.merge_all()

    # # Set up a saver:
    train_writer = tf.summary.FileWriter(LOGDIR)

    other_vars = []
    # print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for n in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):        
        if n not in conv_names:
            other_vars.append(n)
            
    # print other_vars
    

    print "Initialize session ..."
    with tf.Session() as sess:
        
        print 
        print "Train params restore is " + str(train_params['RESTORE'])
        print 
        
        if not train_params['RESTORE']:
            # Restore only conv net weights
            train_writer.add_graph(sess.graph)
            saver = tf.train.Saver(var_list = conv_names)
            saver.restore(sess, conv_params.network_params()['restore_file'])
            # Set up a saver for ALL variables:
            all_saver = tf.train.Saver()
            # Initialize other variables:
            sess.run(tf.variables_initializer(other_vars))
        else: 
            latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
            print latest_checkpoint
            all_saver = tf.train.Saver()
            all_saver.restore(sess, latest_checkpoint)

        print "Begin training ..."
        # Run training loop
        # while not sv.should_stop():
        step = 0
        loader = image_loader("VOC2007")

        while step < train_params['ITERATIONS']:
        # for i in xrange(50):
            step = sess.run(global_step)

            data, labels, boxes = loader.get_next_train_batch(1)
            labels = numpy.squeeze(labels)
            boxes = numpy.squeeze(boxes)
            labels, ground_truths, matched_anchors = rpn_utils.numpy_select_label_anchors_minmax(boxes, _anchors)

            # pos_box_ind, pos_true_ind, neg_box_ind, neg_true_ind = sess.run([
            #     pos_box_ind, pos_true_ind, neg_box_ind, neg_true_ind], 
            #     feed_dict={data_tensor : data,
            #                label_tensor : labels,
            #                box_label : boxes,
            #                # regressor : _fake_regressor, 
            #                anchors : _anchors})

            summary, _ = sess.run(
                [merged_summary, train_step], 
                feed_dict={data_tensor : data,
                           label_tensor : labels,
                           box_label : boxes,
                           # regressor : _fake_regressor, 
                           anchors : _anchors})


            
            # print training accuracy every 10 steps:
            # if i % 10 == 0:
            #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary],
            #                                                       feed_dict={data_tensor:data,
            #                                                                  label_tensor:label})
            #     train_writer.add_summary(loss_s,i)
            #     train_writer.add_summary(accuracy_s,i)

                # sys.stdout.write('Training in progress @ step %d accuracy %g\n' % (i,training_accuracy))
                # sys.stdout.flush()
            
            # Save the model out:
            if step != 0 and step % train_params['SAVE_ITERATION'] == 0:
                all_saver.save(
                    sess,
                    LOGDIR+"/checkpoints/save",
                    global_step=step)


            # if step != 0 and step % 5 == 0:
                # print "Running Summary"
                # _, summ = sess.run([merged_summary], feed_dict={data_tensor: data, 
            #                                      label_tensor: label})
            #     print "Saving Summary"
            #     sv.summary_computed(sess, summ)
            # [loss, acc, _, summary] = sess.run([cross_entropy, accuracy, train_step, merged_summary], 
            #                           feed_dict={data_tensor: data, 
            #                                      label_tensor: label})
            train_writer.add_summary(summary, step)



            # # train_writer.add_summary(summary, i)
            if step % 50 == 0:
                print 'Training in progress @ step ' + str(step)

        # print "\nFinal training loss {}, accuracy {}".format(loss, acc)
        # data, label = mnist.test.next_batch(2000)
        
        # [l, a, summary] = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
        #         data_tensor: data, label_tensor: label})
        # print "\nTesting loss {}, accuracy {}".format(l, a)
