import tensorflow as tf
import numpy
import sys
from models import rpn, rpn_utils, resnet, training_params

from voc_data_loader import voc_meta, image_loader

# Use a resnet for the convolutional step:
conv_params = resnet.resnet_params()
conv_params.network_params()['n_blocks'] = 8
conv_params.network_params()['include_final_classifier'] = True
conv_params.network_params()['n_classes'] = 20
conv_params.network_params()['n_initial_filters'] = 12
conv_params.network_params()['downsample_interval'] = 2
conv_params.network_params()['initial_stride'] = 2
conv_params.network_params()['initial_kernel'] = 5
conv_params.network_params()['bottleneck'] = False
conv_params.network_params()['weight_decay'] = 1E-3
conv_params.network_params()['activation'] = 'softmax'


train_params = dict()
train_params['LOGDIR'] = "logs/rpn_resnet/"
train_params['ITERATIONS'] = 10000
train_params['SAVE_ITERATION'] = 50
train_params['RESTORE'] = False
train_params['RESTORE_INDEX'] = -1
train_params['LEARNING_RATE'] = 0.0001
train_params['DECAY_STEP'] = 100
train_params['DECAY_RATE'] = 0.99
train_params['BATCH_SIZE'] = 12

# Set up the graph:
with tf.Graph().as_default():


    N_MAX_TRUTH = 10
    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [None, 512,512,3], name='x')
    label_tensor = tf.placeholder(tf.float32, [None, 20], name='labels')

    # Let the convolutional part of the network be independant
    # of the classifiers:
    
    conv_net = resnet.resnet(conv_params)
    logits = conv_net.build_network(input_tensor=data_tensor,
                                              is_training=True)    


    # Add cross entropy (loss)
    with tf.name_scope("cross_entropy") as scope:
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits))
        loss_summary = tf.summary.scalar("Loss", cross_entropy)



    LOGDIR = train_params['LOGDIR'] + "/" + conv_net.full_name() + "/"


    # Add a global step accounting for saving and restoring training:
    with tf.name_scope("global_step") as scope:
        global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')

    
    # Add accuracy:
    with tf.name_scope("accuracy") as scope:
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summary = tf.summary.scalar("Accuracy", accuracy)
    
    # Set up a learning rate so we can adapt it:
    learning_rate = tf.train.exponential_decay(learning_rate=train_params['LEARNING_RATE'], 
                                               global_step=global_step,
                                               decay_steps=train_params['DECAY_STEP'],
                                               decay_rate=train_params['DECAY_RATE'],
                                               staircase=True)
    lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
    # Set up a training algorithm:
    with tf.name_scope("training") as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(train_params['LEARNING_RATE'])
            train_step = optimizer.minimize(cross_entropy, global_step=global_step)




    merged_summary = tf.summary.merge_all()

    # # Set up a saver:
    train_writer = tf.summary.FileWriter(LOGDIR)


    print "Initialize session ..."
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        if not train_params['RESTORE']:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            saver = tf.train.Saver()
        else: 
            latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
            print latest_checkpoint
            saver = tf.train.Saver()
            saver.restore(sess, latest_checkpoint)


        print "Begin training ..."
        # Run training loop
        # while not sv.should_stop():
        step = 0
        loader = image_loader("./")

        while step < train_params['ITERATIONS']:
        # for i in xrange(50):
            step = sess.run(global_step)

            data, labels, boxes = loader.get_next_train_batch(train_params['BATCH_SIZE'], mode="class")
            labels = numpy.squeeze(labels)

            # pos_box_ind, pos_true_ind, neg_box_ind, neg_true_ind = sess.run([
            #     pos_box_ind, pos_true_ind, neg_box_ind, neg_true_ind], 
            #     feed_dict={data_tensor : data,
            #                label_tensor : labels,
            #                box_label : boxes,
            #                # regressor : _fake_regressor, 
            #                anchors : _anchors})

            summary, acc, loss, _ = sess.run(
                [merged_summary, accuracy, cross_entropy, train_step], 
                feed_dict={data_tensor : data,
                           label_tensor : labels,
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
            sys.stdout.write(
                'Training in progress @ step %d\r' % (step))
            sys.stdout.flush()

        # print "\nFinal training loss {}, accuracy {}".format(loss, acc)
        # data, label = mnist.test.next_batch(2000)
        sys.stdout.flush()
        
        # [l, a, summary] = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
        #         data_tensor: data, label_tensor: label})
        # print "\nTesting loss {}, accuracy {}".format(l, a)
