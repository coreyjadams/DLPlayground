# Import the mnist data set:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
from models import rpn, rpn_params

from voc_data_loader import voc_meta, image_loader

# mnist data is 28x28 images (784 pixels)

# Set up the network we want to test:
params = rpn_params()
params.network_params()['n_blocks'] = 8
params.network_params()['include_fully_connected'] = False
params.network_params()['n_initial_filters'] = 12
params.network_params()['downsample_interval'] = 2
params.network_params()['initial_stride'] = 2
params.network_params()['initial_kernel'] = 5
params.network_params()['bottleneck'] = False
params.network_params()['weight_decay'] = 1E-3
params.network_params()['activation'] = 'softmax'


params.training_params()['base_lr'] = 1E-3
params.training_params()['lr_decay'] = 0.99
params.training_params()['decay_step']=10

params.training_params()['LOGDIR'] = "logs/rpn/"
params.training_params()['ITERATIONS'] = 5000
params.training_params()['SAVE_ITERATION'] = 100
params.training_params()['RESTORE'] = False
params.training_params()['RESTORE_INDEX'] = -1


# Set up the graph:
with tf.Graph().as_default():



    # Set input data and label for training
    data_tensor = tf.placeholder(tf.float32, [None, 512,512,3], name='x')
    label_tensor = tf.placeholder(tf.float32, [None, 20], name='labels')
    box_label = tf.placeholder(tf.float32, )


    RPN = rpn(params)
    LOGDIR = params.training_params()['LOGDIR'] + "/" + RPN.full_name() + "/"
    print RPN.full_name()

    # if not params.training_params()['RESTORE']:
    classifier, regressor = RPN.build_network(input_tensor=data_tensor, 
                                              n_output_classes=10,
                                              is_training=True)




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



    print "Initialize session ..."
    with tf.Session() as sess:
        
        train_writer.add_graph(sess.graph)
        exit()

        if not params.training_params()['RESTORE']:
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
        while step < params.training_params()['ITERATIONS']:
        # for i in xrange(500):
            step = sess.run(global_step)

            # Receive data (this will hang if IO thread is still running = this
            # will wait for thread to finish & receive data)
            data, label = mnist.train.next_batch(32)


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
            [loss, acc, _, summary] = sess.run([cross_entropy, accuracy, train_step, merged_summary], 
                                      feed_dict={data_tensor: data, 
                                                 label_tensor: label})
            train_writer.add_summary(summary, step)



            # train_writer.add_summary(summary, i)
            sys.stdout.write(
                'Training in progress @ step %d, loss %g, accuracy %g\r' % (step, loss, acc))
            # sys.stdout.flush()

        print "\nFinal training loss {}, accuracy {}".format(loss, acc)
        data, label = mnist.test.next_batch(2000)
        
        [l, a, summary] = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={
                data_tensor: data, label_tensor: label})
        print "\nTesting loss {}, accuracy {}".format(l, a)
