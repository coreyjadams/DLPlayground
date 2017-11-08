# Basic imports
import os,sys,time
import numpy as np

# Import more libraries (after configuration is validated)
import tensorflow as tf
from larcv.dataloader2 import larcv_threadio

from models import resnet3d

# Use a resnet for the convolutional step:
conv_params = resnet3d.resnet3d_params()
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
train_params['RESTORE'] = False
train_params['RESTORE_INDEX'] = -1
train_params['LEARNING_RATE'] = 0.0001
train_params['DECAY_STEP'] = 100
train_params['DECAY_RATE'] = 0.99
train_params['BATCH_SIZE'] = 10

LOGDIR=train_params['LOGDIR']

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

print train_io.fetch_data('label').data()
print train_io.fetch_data('image').dim()
print train_io.fetch_data('image').data().shape


val_io = larcv_threadio()        # create io interface 
train_io_cfg = {'filler_name' : 'Val',
                'verbosity'   : 0, 
                'filler_cfg'  : 'val.cfg'}
val_io.configure(train_io_cfg)   # configure
val_io.start_manager(train_params['BATCH_SIZE']) # start read thread
time.sleep(2)
# retrieve data dimensions to define network later
val_io.next()
dim_data  = val_io.fetch_data('image').dim()
dim_label = val_io.fetch_data('label').dim()

print val_io.fetch_data('label').data()
print val_io.fetch_data('image').dim()
print val_io.fetch_data('image').data().shape



# 2) Configure global process (session, summary, etc.)
#


print "Configuration complete, building network..."

with tf.Graph().as_default():

    # Set input data and label for training
    # Reshape using tensor flow, so expecting a big 1D image as the input:
    data_tensor    = tf.placeholder(tf.float32, 
                                    [None, 
                                     dim_data[1], 
                                     dim_data[2],
                                     dim_data[3],
                                     dim_data[4]],
                                    name='x')
    label_tensor   = tf.placeholder(tf.float32, 
                                    [None, 
                                     2],
                                    name='labels')


    # Initialize the parameters specific to resnet, based on above settings
    ResNet3D = resnet3d.resnet3d(conv_params)

    # Build the network:
    with tf.variable_scope("conv_network"):
        logits = ResNet3D.build_network(input_tensor=data_tensor, n_output_classes=2,
                                        is_training=True)

    LOGDIR += ResNet3D.full_name()
    
    # Add a global step accounting for saving and restoring training:
    with tf.name_scope("global_step") as scope:
        global_step =  tf.Variable(0, dtype=tf.int32, trainable = False, name='global_step')

    # Add cross entropy (loss)
    with tf.name_scope("cross_entropy") as scope:
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits))
        loss_summary = tf.summary.scalar("Loss", cross_entropy)

    # Add accuracy:
    with tf.name_scope("accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summary = tf.summary.scalar("Accuracy", accuracy)

    # # Set up a learning rate so we can adapt it:
    # learning_rate = tf.train.exponential_decay(learning_rate=train_params['base_lr'], 
    #                                            global_step=global_step,
    #                                            decay_steps=params.training_params()['decay_step'],
    #                                            decay_rate=params.training_params()['lr_decay'],
    #                                            staircase=True)
    
    # lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
    # Set up a training algorithm:
    with tf.name_scope("training") as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(train_params['LEARNING_RATE']).minimize(
                cross_entropy, global_step=global_step)



    print "Setting up tensorboard writer ... "
    train_writer = tf.summary.FileWriter(LOGDIR + "/")

    

    merged_summary = tf.summary.merge_all()

    with tf.name_scope("Validation") as scope:
        val_a_summary = tf.summary.scalar("Accuracy", accuracy)
        val_l_summary = tf.summary.scalar("Loss", cross_entropy)

    val_summary = tf.summary.merge([val_a_summary, val_l_summary])


    print "Initialize session ..."
    with tf.Session() as sess:

        if not train_params['RESTORE']:
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            saver = tf.train.Saver()
        else: 
            latest_checkpoint = tf.train.latest_checkpoint(LOGDIR+"/checkpoints/")
            saver = tf.train.Saver()
            saver.restore(sess, latest_checkpoint)

    
    

    
        print "Begin training ..."
        # Run training loop
        step = 0
        while step < train_params['ITERATIONS']:
            step = sess.run(global_step)

                
            # Receive data (this will hang if IO thread is still running = this
            # will wait for thread to finish & receive data)
            batch_data  = train_io.fetch_data('image').data()
            batch_label = train_io.fetch_data('label').data()

            batch_data = np.reshape(batch_data, (batch_data.shape[0], dim_data[1], dim_data[2], dim_data[3], 1))



            # Start IO thread for the next batch while we train the network
            train_io.next()

            # Save the model out:
            if step != 0 and step % train_params['SAVE_ITERATION'] == 0:
                saver.save(
                    sess,
                    LOGDIR+"/checkpoints/save",
                    global_step=step)

            

            # # Test the model with the validation set:
            # if step != 0 and step % TEST_ITERATION == 0:
            #     val_data  = train_proc.fetch_data(STORAGE_KEY_DATA).data()
            #     val_label = train_proc.fetch_data(STORAGE_KEY_LABEL).data()
            #     val_proc.next()
            #     val_data = np.reshape(val_data, (val_data.shape[0], dim_data[1], dim_data[2], dim_data[3], 1))
            #     [v_s] = sess.run([val_summary], 
            #                     feed_dict={data_tensor: val_data, 
            #                                label_tensor : val_label})
            #     train_writer.add_summary(v_s, step)
  
            # Run the training step:
            [l, a, summary, _] = sess.run([cross_entropy, accuracy, merged_summary, train_step], 
                                            feed_dict={data_tensor: batch_data, label_tensor: batch_label})
            #Add this step to the training summary:
            train_writer.add_summary(summary, step)

            sys.stdout.write(
                'Training in progress @ step %d, loss %g, accuracy %g\r' % (step, l, a))
            sys.stdout.flush()
        
        
print "Finished Training."



  
