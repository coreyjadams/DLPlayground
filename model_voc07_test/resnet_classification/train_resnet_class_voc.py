import tensorflow as tf
import numpy
import os, sys, time
from models import rpn, rpn_utils, resnet, training_params
from larcv.dataloader2 import larcv_threadio

# Use a resnet for the convolutional step:
conv_params = resnet.resnet_params()
conv_params.network_params()['n_blocks'] = 16
conv_params.network_params()['include_final_classifier'] = False
conv_params.network_params()['n_classes'] = 20
conv_params.network_params()['n_initial_filters'] = 12
conv_params.network_params()['downsample_interval'] = 4
conv_params.network_params()['initial_stride'] = 2
conv_params.network_params()['initial_kernel'] = 5
conv_params.network_params()['bottleneck'] = False
conv_params.network_params()['weight_decay'] = 1E-3
conv_params.network_params()['activation'] = 'softmax'


train_params = dict()
train_params['LOGDIR'] = "logs/rpn_resnet/"
train_params['ITERATIONS'] = 12000
train_params['SAVE_ITERATION'] = 1000
train_params['RESTORE'] = True
train_params['RESTORE_INDEX'] = -1
train_params['LEARNING_RATE'] = 0.0001
train_params['DECAY_STEP'] = 100
train_params['DECAY_RATE'] = 0.99
train_params['BATCH_SIZE'] = 12

#
# IO
#
train_io = larcv_threadio()        # create io interface 
train_io_cfg = {'filler_name' : 'Train',
                'verbosity'   : 0, 
                'filler_cfg'  : 'train.cfg'}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(train_params['BATCH_SIZE']) # start read thread
time.sleep(2)
# retrieve data dimensions to define network later
train_io.next()
dim_data  = train_io.fetch_data('image').dim()
dim_label = train_io.fetch_data('label').dim()

print train_io.fetch_data('image')

exit()

# Set up the graph:
with tf.Graph().as_default():


    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [None, 512,512,3], name='x')
    label_tensor = tf.placeholder(tf.float32, [None, 20], name='labels')

    # Let the convolutional part of the network be independant
    # of the classifiers:
    
    with tf.variable_scope("ResNet"):
        conv_net = resnet.resnet(conv_params)
        final_conv_layer = conv_net.build_network(input_tensor=data_tensor,
                                                  is_training=True)    
    
    
    # Only saving the variables that are directly part of the convolutional network:
    # conv_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

    # Map the final conv layers to make statements about the presence 
    # of each class:
    conv_names = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "ResNet")

    n = final_conv_layer.get_shape().as_list()[1]
    
    with tf.variable_scope("Classification-FC"):
        x = tf.layers.conv2d(final_conv_layer,
                             512,
                             kernel_size=[n, n],
                             strides=[1, 1],
                             padding='valid',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None,  # automatically uses Xavier initializer
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             trainable=True,
                             name="Conv2DNxN",
                             reuse=None)

    with tf.variable_scope("Classification-Final"):
        classifier = tf.layers.conv2d(x,
                                     20,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=None,  # automatically uses Xavier initializer
                                     kernel_regularizer=None,
                                     activity_regularizer=None,
                                     trainable=True,
                                     name="Conv2D1x1",
                                     reuse=None)

        # Reshape the classifier into the feature map pools it was using:
        classifier = tf.reshape(classifier, (tf.shape(classifier)[0], 20))
        
    # The classifier now has 20 individual classifiers that are predicting the
    # presence of the 20 classes in the image.  

    # classifier = tf.nn.softmax(classifier)

    # Add cross entropy (loss)
    with tf.name_scope("cross_entropy") as scope:
        cross_entropy = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=label_tensor, logits=classifier, pos_weight=0.5*11.5))
        loss_summary = tf.summary.scalar("Loss", cross_entropy)



    LOGDIR = train_params['LOGDIR'] + "/" + conv_net.full_name() + "/"


    # Add a global step accounting for saving and restoring training:
    with tf.name_scope("global_step") as scope:
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    prediction = tf.round(tf.sigmoid(classifier))
    
    # Add accuracy:
    with tf.name_scope("accuracy") as scope:
        correct_prediction = tf.equal(
            prediction, label_tensor)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summary = tf.summary.scalar("Accuracy", accuracy)
    
    # # Set up a learning rate so we can adapt it:
    # learning_rate = tf.train.exponential_decay(learning_rate=train_params['LEARNING_RATE'], 
    #                                            global_step=global_step,
    #                                            decay_steps=train_params['DECAY_STEP'],
    #                                            decay_rate=train_params['DECAY_RATE'],
    #                                            staircase=True)
    # Set up a learning rate so we can adapt it:
    learning_rate = tf.placeholder(tf.float32, ())    
    lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
    # Set up a training algorithm:
    with tf.name_scope("training") as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(train_params['LEARNING_RATE'])
            train_step = optimizer.minimize(cross_entropy, global_step=global_step)

    with tf.name_scope("Validation") as scope:
        val_a_summary = tf.summary.scalar("Accuracy", accuracy)
        val_l_summary = tf.summary.scalar("Loss", cross_entropy)

    val_summary = tf.summary.merge([val_a_summary, val_l_summary])


    merged_summary = tf.summary.merge_all()

    # # Set up a saver:
    train_writer = tf.summary.FileWriter(LOGDIR)


    print "Initialize session ..."
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        
        if not train_params['RESTORE']:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
        else: 
            print LOGDIR
            latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
            print latest_checkpoint
            saver.restore(sess, latest_checkpoint)


        print "Begin training ..."
        # Run training loop
        # while not sv.should_stop():
        step = 0
        loader = image_loader("VOC2007")

        while step < train_params['ITERATIONS']:
        # for i in xrange(5):
            step = sess.run(global_step)

            data, labels, boxes = loader.get_next_train_batch(train_params['BATCH_SIZE'], mode="class")
            # labels = numpy.squeeze(labels)


            summary, acc, loss, _ = sess.run(
                [merged_summary, accuracy, cross_entropy, train_step], 
                feed_dict={data_tensor : data,
                           label_tensor : labels,
                           learning_rate : train_params['LEARNING_RATE']
                           # regressor : _fake_regressor, 
                           })


            # print training accuracy every 10 steps:
            # if i % 10 == 0:
            #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary],
            #                                                       feed_dict={data_tensor:data,
            #                                                                  label_tensor:label})
            #     train_writer.add_summary(loss_s,i)
            #     train_writer.add_summary(accuracy_s,i)

            sys.stdout.write('Training in progress @ step %d accuracy %g, loss %g\n' % (step,acc,loss))
            sys.stdout.flush()
            
            # Save the model out:
            if step != 0 and step % train_params['SAVE_ITERATION'] == 0:
                saver.save(
                    sess,
                    LOGDIR+"/checkpoints/save",
                    global_step=step)
                
                
            
            train_writer.add_summary(summary, step)


            if step != 0 and step % 50 == 0:
                val_data, val_labels, val_boxes = loader.get_next_val_batch(4*train_params['BATCH_SIZE'], mode="class")
                val_sum, = sess.run([val_summary], feed_dict={data_tensor: val_data, 
                                                 label_tensor: val_labels})
                train_writer.add_summary(val_sum, step)
                



            # # train_writer.add_summary(summary, i)
            sys.stdout.write(
                'Training in progress @ step %d\r' % (step))
            sys.stdout.flush()

        # print "\nFinal training loss {}, accuracy {}".format(loss, acc)
        # data, label = mnist.test.next_batch(2000)
        sys.stdout.flush()
        
        # [l, a, summary] = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
        #         data_tensor: data, label_tensor: label})
        # print "\nTesting loss {}, accuracy {}".format(l, a)
