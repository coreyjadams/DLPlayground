# Basic imports
import os,sys,time
from toy_config import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load configuration and check if it's good
cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
  sys.exit(1)

# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

# Import more libraries (after configuration is validated)
import tensorflow as tf
from larcv.dataloader2 import larcv_threadio

#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
if not cfg.FILLER_CONFIG:
  'Must provide larcv data filler configuration file!'
  sys.exit(1)
proc = larcv_threadio()
filler_cfg = {'filler_name': 'ThreadProcessor',
              'verbosity':0, 
              'filler_cfg':cfg.FILLER_CONFIG}
proc.configure(filler_cfg)
# Start IO thread
proc.start_manager(cfg.BATCH_SIZE)
# Storage ID
storage_id=0
# Retrieve image/label dimensions
proc.next()
dim_data    = proc.fetch_data(cfg.STORAGE_KEY_DATA).dim()
dim_label   = proc.fetch_data(cfg.STORAGE_KEY_LABEL).dim()

import constructor
data_tensor,label_tensor,forward,loss,train = constructor.build(cfg.ARCHITECTURE, dim_data, dim_label, train=cfg.TRAIN)

#
# 2) Configure global process (session, summary, etc.)
#
# Create a bandle of summary
merged_summary=tf.summary.merge_all()
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Create a summary writer handle
writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)
# Create weights saver
saver = tf.train.Saver()
# Override variables if wished
if cfg.LOAD_FILE:
  vlist=[]
  avoid_params=cfg.AVOID_LOAD_PARAMS.split(',')
  for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if v.name in avoid_params:
      print '\033[91mSkipping\033[00m loading variable',v.name,'from input weight...'
      continue
    print '\033[95mLoading\033[00m variable',v.name,'from',cfg.LOAD_FILE
    vlist.append(v)
  reader=tf.train.Saver(var_list=vlist)
  reader.restore(sess,cfg.LOAD_FILE)
 
# Run training loop
for i in range(cfg.ITERATIONS):
  
  # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
  batch_data  = proc.fetch_data(cfg.STORAGE_KEY_DATA).data()
  batch_label = proc.fetch_data(cfg.STORAGE_KEY_LABEL).data()

  if cfg.DEBUG:
    for entry in xrange(len(batch_data[0])):
      entry_pydata = batch_data[entry].reshape(dim_data[1:4]).astype(np.float32)
      print 'Entry',entry,
      print 'max',batch_data[entry].max(),
      print 'min',batch_data[entry].min(),
      print 'mean',batch_data[entry].mean(),
      print 'sum',batch_data[entry].sum(),
      print 'ctr',len(entry_pydata.nonzero()[0])

      z,y,x = entry_pydata.nonzero()
      fig = plt.figure()
      ax = fig.add_subplot(111,projection='3d')
      ax.scatter(x,y,z,zdir='z',cmap='jet')
      plt.savefig('Entry%03d.png' % entry)

  # Start IO thread for the next batch while we train the network
  proc.next()
  # Execute
  loss_value,acc_value = (None,None)
  if train:
    loss_value,acc_value,_ = sess.run([loss,forward,train],feed_dict={data_tensor: batch_data, label_tensor: batch_label})
    sys.stdout.write('Training in progress @ step %d loss %g accuracy %g            \r' % (i,loss_value,acc_value))
  else:
    acc_value = sess.run(forward,feed_dict={data_tensor: batch_data, label_tensor: batch_label})
    sys.stdout.write('Training in progress @ step %d accuracy %g                    \r' % (i,acc_value))  

  sys.stdout.flush()

  # Save log every 20 steps
  if train and (i+1)%20 == 0:
    # Run summary
    s = sess.run(merged_summary, feed_dict={data_tensor:batch_data, label_tensor:batch_label})
    writer.add_summary(s,i)
  
  # If configured to save summary + snapshot, do so here.
  if train and (i+1)%cfg.SAVE_ITERATION == 0:
    # Save snapshot
    ssf_path = saver.save(sess,cfg.ARCHITECTURE,global_step=i)
    print
    print 'saved @',ssf_path

  
