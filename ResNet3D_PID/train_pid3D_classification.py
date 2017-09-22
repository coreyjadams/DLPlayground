# Basic imports
import os,sys,time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Classification label conversion
def convert_label(input_label,num_class):
  result_label = np.zeros((len(input_label),num_class))
  for idx,label in enumerate(input_label):
    result_label[idx][int(label)]=1.
  return result_label

# Import more libraries (after configuration is validated)
import tensorflow as tf
from larcv.dataloader2 import larcv_threadio

from models import resnet3d


# Set up the network we want to test:
params = resnet3d.resnet3d_params()
params.network_params()['n_blocks'] = 16
params.network_params()['include_fully_connected'] = False
params.network_params()['n_initial_filters'] = 24
params.network_params()['downsample_interval'] = 4
params.network_params()['initial_stride'] = 2
params.network_params()['initial_kernel'] = 5
params.network_params()['bottleneck'] = False
params.network_params()['weight_decay'] = 1E-3
params.network_params()['activation'] = 'softmax'


params.training_params()['NUM_CLASS'] = 5
params.training_params()['BATCH'] = 6
params.training_params()['TEST_BATCH'] = 48
params.training_params()['LOGDIR'] = "logs/"
params.training_params()['ITERATIONS'] = 50000
params.training_params()['SAVE_ITERATION'] = 1000
params.training_params()['TEST_ITERATION'] = 50
params.training_params()['RESTORE'] = False

params.training_params()['base_lr'] = 1E-4
params.training_params()['lr_decay'] = 0.99
params.training_params()['decay_step']=100


LOGDIR = params.training_params()['LOGDIR']
ITERATIONS = params.training_params()['ITERATIONS']
SAVE_ITERATION = params.training_params()['SAVE_ITERATION']
TEST_ITERATION = params.training_params()['TEST_ITERATION']


STORAGE_KEY_DATA="BatchFillerVoxel3D"
STORAGE_KEY_LABEL="BatchFillerPIDLabel"

BATCH_SIZE = params.training_params()['BATCH']

#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure

train_proc = larcv_threadio()
filler_cfg = {'filler_name': 'ThreadProcessorTrain',
              'verbosity':0, 
              'filler_cfg':"pid3d_train.cfg"}
train_proc.configure(filler_cfg)
# Start IO thread
train_proc.start_manager(BATCH_SIZE)
# Storage ID
storage_id=0
# Retrieve image/label dimensions
train_proc.next()
dim_data    = train_proc.fetch_data(STORAGE_KEY_DATA).dim()
dim_label   = train_proc.fetch_data(STORAGE_KEY_LABEL).dim()



val_proc = larcv_threadio()
filler_cfg = {'filler_name': 'ThreadProcessorVal',
              'verbosity':0, 
              'filler_cfg':"pid3d_val.cfg"}
val_proc.configure(filler_cfg)
# Start IO thread
val_proc.start_manager(params.training_params()['TEST_BATCH'])
# Storage ID
storage_id=0
# Retrieve image/label dimensions
val_proc.next()



print dim_data
print dim_label

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
                                     1],
                                    name='x')
    label_tensor   = tf.placeholder(tf.float32, 
                                    [None, 
                                     params.training_params()['NUM_CLASS']],
                                    name='labels')


    # Initialize the parameters specific to resnet, based on above settings
    ResNet3D = resnet3d.resnet3d(params)

    # Build the network:
    logits = ResNet3D.build_network(input_tensor=data_tensor, n_output_classes=5,
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

    # Set up a learning rate so we can adapt it:
    learning_rate = tf.train.exponential_decay(learning_rate=params.training_params()['base_lr'], 
                                               global_step=global_step,
                                               decay_steps=params.training_params()['decay_step'],
                                               decay_rate=params.training_params()['lr_decay'],
                                               staircase=True)

    lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
    # Set up a training algorithm:
    with tf.name_scope("training") as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(0.1*learning_rate).minimize(
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

        if not params.training_params()['RESTORE']:
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
        while step < params.training_params()['ITERATIONS']:
            step = sess.run(global_step)

                
            # Receive data (this will hang if IO thread is still running = this
            # will wait for thread to finish & receive data)
            batch_data  = train_proc.fetch_data(STORAGE_KEY_DATA).data()
            batch_label = train_proc.fetch_data(STORAGE_KEY_LABEL).data()

            batch_data = np.reshape(batch_data, (batch_data.shape[0], dim_data[1], dim_data[2], dim_data[3], 1))



            # Start IO thread for the next batch while we train the network
            train_proc.next()

            # Save the model out:
            if step != 0 and step % params.training_params()['SAVE_ITERATION'] == 0:
                saver.save(
                    sess,
                    LOGDIR+"/checkpoints/save",
                    global_step=step)

            

            # Test the model with the validation set:
            if step != 0 and step % TEST_ITERATION == 0:
                val_data  = train_proc.fetch_data(STORAGE_KEY_DATA).data()
                val_label = train_proc.fetch_data(STORAGE_KEY_LABEL).data()
                val_proc.next()
                val_data = np.reshape(val_data, (val_data.shape[0], dim_data[1], dim_data[2], dim_data[3], 1))
                [v_s] = sess.run([val_summary], 
                                feed_dict={data_tensor: val_data, 
                                           label_tensor : val_label})
                train_writer.add_summary(v_s, step)
  
            # Run the training step:
            [l, a, summary, _] = sess.run([cross_entropy, accuracy, merged_summary, train_step], 
                                            feed_dict={data_tensor: batch_data, label_tensor: batch_label})
            #Add this step to the training summary:
            train_writer.add_summary(summary, step)

            sys.stdout.write(
                'Training in progress @ step %d, loss %g, accuracy %g\r' % (step, l, a))
            sys.stdout.flush()
        
        
print "Finished Training."



  
