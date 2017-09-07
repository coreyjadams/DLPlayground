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

from models import resnet

# Set up the network we want to test:
params = resnet.resnet_params()
params.network_params()['n_blocks'] = 20
params.network_params()['include_fully_connected'] = False
params.network_params()['n_initial_filters'] = 32
params.network_params()['downsample_interval'] = 5
params.network_params()['initial_stride'] = 2
params.network_params()['initial_kernel'] = 5
params.network_params()['bottleneck'] = False
params.network_params()['weight_decay'] = 1E-3
params.network_params()['activation'] = 'softmax'


params.training_params()['base_lr'] = 5E-3
params.training_params()['lr_decay'] = 0.96
params.training_params()['decay_step']=100


# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

# Import more libraries (after configuration is validated)
import numpy as np
import tensorflow as tf
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
train_proc = larcv_data()
filler_cfg = {'filler_name': 'DataFillerTrain', 
              'verbosity':0, 
              'filler_cfg':'%s/ResNet_PID/oneclass_train_filler.cfg' % os.environ['DLPLAYGROUND']}
train_proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
train_proc.read_next(cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
input, labels = train_proc.next()
# Immediately start the thread for later IO
train_proc.read_next(cfg.BATCH_SIZE)
# Retrieve image/label dimensions
image_dim = train_proc.image_dim()
label_dim = train_proc.label_dim()


val_proc = larcv_data()
filler_cfg = {'filler_name': 'DataFillerVal', 
              'verbosity':0, 
              'filler_cfg':'%s/ResNet_PID/oneclass_val_filler.cfg' % os.environ['DLPLAYGROUND']}
val_proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
val_proc.read_next(4*cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
input, labels = val_proc.next()
# Immediately start the thread for later IO
val_proc.read_next(4*cfg.BATCH_SIZE)


#
# 1) Build network
#

print "Configuration complete, building network..."


# Set input data and label for training
data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2]*image_dim[3]],name='x')
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')

# Reshape the tensor to be 512 x 512:
x = tf.reshape(data_tensor, (tf.shape(data_tensor)[0], 512, 512, 1))

ResNet = resnet.resnet(params)

logits = ResNet.build_network(input_tensor=x, n_output_classes=5,
                              is_training=True)


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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy, global_step=global_step)



print "Setting up tensorboard writer ... "

LOGDIR = cfg.LOGDIR
ITERATIONS = 20
SAVE_ITERATION = 5
TEST_ITERATION = 3

train_writer = tf.summary.FileWriter(LOGDIR + "/" + ResNet.full_name() + "/")
# snapshot_writer = tf.summary.FileWriter(LOGDIR + "/snapshot/")
saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()

with tf.name_scope("Validation") as scope:
    val_a_summary = tf.summary.scalar("Accuracy", accuracy)
    val_l_summary = tf.summary.scalar("Loss", cross_entropy)

val_summary = tf.summary.merge([val_a_summary, val_l_summary])

print "Initialize session ..."
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)

    print "Begin training ..."
    # Run training loop
    for i in range(ITERATIONS):

        # Receive data (this will hang if IO thread is still running = this
        # will wait for thread to finish & receive data)
        data,label = train_proc.next()
        # Start IO thread for the next batch while we train the network
        train_proc.read_next(cfg.BATCH_SIZE)
        # Use utility function to convert the shape of the label for classification
        label = convert_label(label,cfg.NUM_CLASS)

        print data.shape
        print label.shape

        if i != 0 and i % SAVE_ITERATION == 0:
            saver.save(
                sess,
                LOGDIR+"/checkpoints/densenet_pid_{}".format(ResNet.full_name()),
                global_step=global_step)

        if i % TEST_ITERATION == 0:
            val_data, val_label = val_proc.next()
            val_proc.read_next(4*cfg.BATCH_SIZE)
            val_label = convert_label(val_label,cfg.NUM_CLASS)
            [v_s] = sess.run([val_summary], 
                            feed_dict={data_tensor: val_data, label_tensor : val_label})
            train_writer.add_summary(v_s, i)
        # print training accuracy every 10 steps:
        # if i % 10 == 0:
        #     training_accuracy, loss_s, accuracy_s, = sess.run([accuracy, loss_summary, acc_summary],
        #                                                       feed_dict={data:data,
        #                                                                  label:label})
        #     train_writer.add_summary(loss_s,i)
        #     train_writer.add_summary(accuracy_s,i)

            # sys.stdout.write('Training in progress @ step %d accuracy %g\n' % (i,training_accuracy))
            # sys.stdout.flush()

        [l, a, summary, _] = sess.run([cross_entropy, accuracy, merged_summary, train_step], 
                                        feed_dict={data_tensor: data, label_tensor: label})
        train_writer.add_summary(summary, i)
        sys.stdout.write(
            'Training in progress @ step %d, loss %g, accuracy %g\r' % (i, l, a))
        sys.stdout.flush()