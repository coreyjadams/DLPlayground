# Basic imports
import os,sys,time

from models import resnet

# Set up the network we want to test:
params = resnet.resnet_params()
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
params.training_params()['BATCH'] = 24
params.training_params()['TEST_BATCH'] = 48
params.training_params()['LOGDIR'] = "logs/"
params.training_params()['ITERATIONS'] = 1416/2
params.training_params()['SAVE_ITERATION'] = 1000
params.training_params()['TEST_ITERATION'] = 50
params.training_params()['RESTORE'] = True

params.training_params()['base_lr'] = 5E-4
params.training_params()['lr_decay'] = 0.99
params.training_params()['decay_step']=100

LOGDIR = params.training_params()['LOGDIR']
ITERATIONS = params.training_params()['ITERATIONS']



# Import more libraries (after configuration)
import numpy as np
import tensorflow as tf
from utils.dataloader import larcv_data

#
# Utility functions
#

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
filler_cfg = {'filler_name': 'DataFillerTest', 
              'verbosity':0, 
              'filler_cfg':'%s/ResNet_PID/oneclass_test_filler.cfg' % os.environ['DLPLAYGROUND']}
train_proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
train_proc.read_next(params.training_params()['BATCH'])
# Force data to be read (calling next will sleep enough for the IO thread to finish reading)
input, labels = train_proc.next()
# Immediately start the thread for later IO
# train_proc.read_next(params.training_params()['BATCH'])
# Retrieve image/label dimensions
image_dim = train_proc.image_dim()
label_dim = train_proc.label_dim()


from larcv import larcv  
from ROOT import TChain, TTree, TFile
from array import array
import os

filler = larcv.ThreadFillerFactory.get_filler("DataFillerTest")  
roi_chain = TChain("partroi_segment_tree")  
for fname in filler.pd().io().file_list():    
    roi_chain.AddFile(fname)  

import time
while filler.thread_running():
    time.sleep(0.1)
    
filler.set_next_index(0)
# train_proc.read_next(params.training_params()['BATCH'])
    

#
# 1) Build network
#

print "Configuration complete, building network..."

with tf.Graph().as_default():

    # Set input data and label for training
    # Reshape using tensor flow, so expecting a big 1D image as the input:
    data_tensor    = tf.placeholder(tf.float32, 
                                    [None, 
                                     image_dim[2]*image_dim[3]],
                                    name='x')
    label_tensor   = tf.placeholder(tf.float32, 
                                    [None, 
                                     params.training_params()['NUM_CLASS']],
                                    name='labels')

    # Reshape the tensor to be 512 x 512:
    x = tf.reshape(data_tensor, (tf.shape(data_tensor)[0], 512, 512, 1))

    # Initialize the parameters specific to resnet, based on above settings
    ResNet = resnet.resnet(params)

    # Build the network:
    logits = ResNet.build_network(input_tensor=x, n_output_classes=5,
                                  is_training=True)

    LOGDIR += ResNet.full_name()
    
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

#     # Set up a learning rate so we can adapt it:
#     learning_rate = tf.train.exponential_decay(learning_rate=params.training_params()['base_lr'], 
#                                                global_step=global_step,
#                                                decay_steps=params.training_params()['decay_step'],
#                                                decay_rate=params.training_params()['lr_decay'],
#                                                staircase=True)

#     lr_summary = tf.summary.scalar("Learning Rate", learning_rate)
    
#     # Set up a training algorithm:
#     with tf.name_scope("training") as scope:
#         train_step = tf.train.AdamOptimizer(learning_rate).minimize(
#             cross_entropy, global_step=global_step)



    # print "Setting up tensorboard writer ... "
#     train_writer = tf.summary.FileWriter(LOGDIR + "/")

    

#     merged_summary = tf.summary.merge_all()

#     with tf.name_scope("Validation") as scope:
#         val_a_summary = tf.summary.scalar("Accuracy", accuracy)
#         val_l_summary = tf.summary.scalar("Loss", cross_entropy)

#     val_summary = tf.summary.merge([val_a_summary, val_l_summary])

    print "Initialize session ..."
    with tf.Session() as sess:

        # print "LOGDIR is " + LOGDIR
        
        latest_checkpoint = tf.train.latest_checkpoint(LOGDIR + "/checkpoints/")
        print latest_checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, latest_checkpoint)

        
        # Set up an output ttree to contain the results
        # Run/subrun/event
        # Particle PDG
        # Particle Energy
        # Particle direction, x y z
        # Classification scores for each category
        fname = "analysis_" + ResNet.full_name() + "_" +  os.path.basename(latest_checkpoint) + ".root"
        f = TFile(fname, 'recreate')
        t = TTree('analysis', "PID analysis tree")
        pdg = array('i', [0])
        px = array('f', [0])
        py = array('f', [0])
        pz = array('f', [0])
        e  = array('f', [0])
        run = array('i', [0])
        subrun = array('i', [0])
        event = array('i', [0])
        larcv_label = array('i', [0])
        logit_ar = array('f', 5*[0])
        selection = array('f', [0])
    
        t.Branch("pdg", pdg, "pdg/I")
        t.Branch("px", px, "px/F")
        t.Branch("py", py, "py/F")
        t.Branch("pz", pz, "pz/F")
        t.Branch("e", e, "e/F")
        t.Branch("run", run, "run/I")
        t.Branch("subrun", subrun, "subrun/I")
        t.Branch("event", event, "event/I")
        t.Branch("label", larcv_label, "label/I")
        t.Branch("logit_ar", logit_ar, "logit_ar[5]/F")
        t.Branch("selection", selection, "selection/F")
    
        print "Begin analysis ..."
        # Run training loop
        step = 0
        cum_acc = 0
        
        while step < params.training_params()['ITERATIONS']:
            step += 1

            print "On batch {}".format(step)

            # Receive data (this will hang if IO thread is still running = this
            # will wait for thread to finish & receive data)
            
            # Start IO thread for the next batch while we train the network
            train_proc.read_next(params.training_params()['BATCH'])
            data,orig_label = train_proc.next()

            # Use utility function to convert the shape of the label for classification
            label = convert_label(orig_label,params.training_params()['NUM_CLASS'])

            
            # Run the training step:
            [log, loss, acc] = sess.run([logits, cross_entropy, accuracy], 
                                        feed_dict={data_tensor: data, label_tensor: label})

            i = 0
            for entry in filler.processed_entries():
                roi_chain.GetEntry(entry)
                br = roi_chain.partroi_segment_branch
                pdg[0] = br.at(0).PdgCode()
                px[0] = br.at(0).Px()
                py[0] = br.at(0).Py()
                pz[0] = br.at(0).Pz()
                e [0] = br.at(0).EnergyInit()
                run[0] = br.run()
                subrun[0] = br.subrun()
                event[0] = br.event()
                larcv_label[0] = orig_label[i]
                selection[0] = np.argmax(log[i])
                for j in xrange(5):
                    logit_ar[j] = log[i][j]
                t.Fill()
                i += 1
            
            cum_acc += acc
            # print log

f.Write()
f.Close()
    
print "Finished Evaluation."
print "Cumulative accuracy: " + str(cum_acc / step)
