# Basic imports
import os,sys,time
from utils import config

# Load configuration and check if it's good
cfg = config()
_cfg = None
for _f in sys.argv:
    if ".cfg" in _f:
        _cfg = _f
        break
 

if not cfg.parse(_cfg) or not cfg.sanity_check():
  sys.exit(1)




# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

# Import more libraries (after configuration is validated)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from utils.dataloader import larcv_data

#
# Utility functions
#
# Integer rounder
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

# Classification label conversion
def convert_label(input_label,num_class):
  result_label = np.zeros((len(input_label),num_class))
  for idx,label in enumerate(input_label):
    result_label[idx][int(label)]=1.
  return result_label


#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
proc = larcv_data()
filler_cfg = {'filler_name': 'DataFiller', 
              'verbosity':0, 
              'filler_cfg':'%s/DenseNet_PID/oneclass_filler.cfg' % os.environ['DLPLAYGROUND']}
proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
proc.read_next(cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
input, labels = proc.next()
# Immediately start the thread for later IO
proc.read_next(cfg.BATCH_SIZE)
# Retrieve image/label dimensions
image_dim = proc.image_dim()
label_dim = proc.label_dim()


#
# 1) Build network
#

print "Configuration complete, building network..."


# Set input data and label for training
data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] , image_dim[3], 1],name='x')
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')


# import model building functions
from models.densenet import densenet

DenseNet = densenet()

EXTRA_NAME="lr_1e-2_nblocks_5_nlpb_5_gr24_initstride_2"

logits = DenseNet.build_dense_net(input_tensor=data_tensor, n_output_classes=5, n_blocks=5, n_layers_per_block=5,
                                 include_fully_connected = False, growth_rate=24, is_training=True,
                                 n_initial_filters=32, initial_stride=2, bottleneck=True, compression_factor=0.5, 
                                 dropout_rate=0.5, weight_decay=1e-4, activation='softmax')



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

# Set up a training algorithm:
with tf.name_scope("training") as scope:
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, global_step=global_step)



print "Setting up tensorboard writer ... "

train_writer = tf.summary.FileWriter(cfg.LOGDIR + "/" + EXTRA_NAME +"/")
# snapshot_writer = tf.summary.FileWriter(cfg.LOGDIR + "/snapshot/")
saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
print type(merged_summary)

print "Initialize session ..."
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)


    print "Begin training ..."
    # Run training loop
    for i in range(cfg.ITERATIONS):

        # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
        data,label = proc.next()
        # Start IO thread for the next batch while we train the network
        proc.read_next(cfg.BATCH_SIZE)
        # Use utility function to convert the shape of the label for classification
        label = convert_label(label,cfg.NUM_CLASS)
        # Run loss & train step
        data = np.reshape(data, (cfg.BATCH_SIZE, image_dim[2], image_dim[3], 1))

        if i-1 % cfg.SAVE_ITERATION == 0:
            saver.save(sess, cfg.LOGDIR+"/checkpoints/densenet_pid_{}".format(EXTRA_NAME), global_step=global_step)
        
        #print training accuracy every 10 steps:
        # if i % 10 == 0:
        #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary], 
        #                                                       feed_dict={data_tensor:data, 
        #                                                                  label_tensor:label})
        #     train_writer.add_summary(loss_s,i)
        #     train_writer.add_summary(accuracy_s,i)
        
            # sys.stdout.write('Training in progress @ step %d accuracy %g\n' % (i,training_accuracy))
            # sys.stdout.flush()
            
        [summary, _] = sess.run([merged_summary, train_step], feed_dict={data_tensor:data, label_tensor:label})
        train_writer.add_summary(summary, i)
        sys.stdout.write('Training in progress @ step %d\r' % (i))
        sys.stdout.flush()
    


