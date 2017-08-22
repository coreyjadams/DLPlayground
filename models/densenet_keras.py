'''
Implementation of densenet for larcv with keras.
Aiming to include both classification and fully convolultional versions
'''

# keras imports:
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf


def _transistion_block(input_tensor, n_filters, 
                       compression_factor, 
                       label="",
                       dropout = None,
                       weight_decay=1e-4):
    '''
    input_tensor is the input keras tensor
    n_filters is the number of filters per stage
    compression_factor is the rate at which to reduce the number of filters, using 1x1 conv
    weight_decay is the weight_decay parameter using L2 normalization
    '''
    with tf.name_scope("transition_{}".format(label)):
        
        # Batch normalization is applied first:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), 
                               beta_regularizer=l2(weight_decay),
                              name="batch_norm_transition_{}".format(label))(input_tensor)

        # ReLU:
        x = Activation('relu')(x)

        # 1x1 convolution to reduce the number of filters
        n_output_filters = int(n_filters*compression_factor)
        x = Conv2D(n_output_filters, kernel_size=(1,1), strides=1, padding="same",
                   use_bias=False,  kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(x)

        if dropout is not None:
            # Apply the dropout rate:
            x = Dropout(dropout)(x)

        # Average pooling 2D:
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)

        return x
    
    
    
    
    
def _convolution_block(input_tensor, n_filters, label="",
                       bottleneck=False, dropout=None, weight_decay=1e-4):
    '''
    input_tensor is the input keras tensor
    n_filters is the number of filters produced at each layer of a dense block
    bottleneck determines if feature reduction is applied before each 3x3 conv
    dropout is the dropout rate
    weight_decay is the regularization term (L2)
    '''
    with tf.name_scope("convolution_{}".format(label)):

        # Batch normalization is applied first:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(input_tensor)

        # ReLU:
        x = Activation('relu')(x)


        if bottleneck:
            # 1x1 convolution to reduce the number of filters
            x = Conv2D(4*n_filters, kernel_size=(1,1), strides=1, padding="same",
                       use_bias=False,  kernel_initializer='he_uniform',
                       kernel_regularizer=l2(weight_decay))(x)

            # Dropout
            if dropout is not None:
                x = Dropout(dropout)(x)

            # Batch Norm
            x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

            # ReLU:
            x = Activation('relu')(x)


        # Apply the 3x3 convolutional layer:
        x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding="same",
                   use_bias=False,  kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay))(x)

        #Dropout
        if dropout is not None:
            x = Dropout(dropout)(x)

        # Finished a convolutional block
        return x

    
def _dense_block(x, n_layers, growth_rate, label="",
                 bottleneck=False, dropout=None, weight_decay=1e-4):
    '''
    x is the input keras_tensor, LxWxK
    n_layers is the number of convolutional layers to apply
    growth_rate is the number of filters that get added at each convolutional block
    bottleneck is whether to apply bottleneck at each convolutional layer
    dropout is the dropout rate
    weight_decay is the rate of weight decay
    '''
    
    print "label is " + str(label)
    
    with tf.name_scope("dense_{}".format(label)):

        # For each convolutional block, concatenate it's output with it's input
        for layer in range(n_layers):
            lab =  "{}_{}".format(label, layer)
            conv_output = _convolution_block(x, growth_rate, lab, bottleneck, dropout, weight_decay)

            x = concatenate([x, conv_output])

        return x


def _build_dense_net(input_tensor, n_output_classes, n_blocks = 3, n_layers_per_block = 4, 
                     include_fully_connected = False, growth_rate=16,
                     n_initial_filters=-1, bottleneck=False, compression_factor=1.0,
                     dropout_rate=None, weight_decay=1e-4, activation='softmax'):
    '''
    Create a densenet model (model initialization done elsewhere)
    Parameters:
        input_tensor - the input image tensor, channels last.  Must be shape (Batch, Length, Width, Channels)
                       (probably, input channel number is 1 or == nViews)
        n_output_classes - this is a classification network, so list the number of desired output classes
        n_blocks - number of dense blocks in the network
        n_layers_per_block - the number of convolutional layers per block
        include_fully_connected - if true, include a fully connected layer at the end of the network
        growth_rate - number of new channels to add at each convolution in the dense blocks
        n_initial_filters - number of initial filters to produce before the first dense block
                            (Defaults to growth_rate * 2)
        bottleneck - if True, bottleneck convolutions are applied in the convolutional blocks to reduce the number of features
        compression_factor - reduce the number of channels (features) in the transition layer with 1x1 convolution, 
                             to reduce the input to the next denseblock.  Output of each dense blocks is 
                             n_initial_filters + n_layers_per_block*growth_rate.  0.5 is used in the paper.
        dropout_rate - the rate at which dropout is applied.  default is none
        weight_decay - regularization parameter for weight decay in the lose function, default is 1e-4
        activation - activation function for the fully connected layer, if used.
    '''
    
    print input_tensor.shape
    
    # Do some basic parameter checking:
    assert compression_factor > 0 and compression_factor <=1, 'Compression must be between 0 and 1.'
    
    if n_initial_filters == -1:
        n_initial_filters = 2*growth_rate
        
    assert n_initial_filters > 0, 'Number of initial filters must be positive.'


    # Now, start building the model

    x = Conv2D(n_initial_filters, (7, 7), strides=(3,3), kernel_initializer='he_uniform',
               padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(input_tensor)
    
    
    # Apply dense blocks and transition blocks alternately, except no transition block after the last dense block:
    for block in xrange(n_blocks - 1):
        
        x = _dense_block(x, n_layers_per_block, growth_rate, label=block,
                         bottleneck=bottleneck, dropout=dropout_rate, 
                         weight_decay=weight_decay)
    
        
        x = _transistion_block(x, 4*growth_rate, compression_factor, 
                               label=block, dropout=dropout_rate, 
                               weight_decay=weight_decay)

                      

    # The last dense_block:
    x = _dense_block(x, n_layers_per_block, growth_rate, label=n_blocks-1, bottleneck=bottleneck,
                     dropout=dropout_rate, weight_decay=weight_decay)

    with tf.name_scope("final_pooling"):
        x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

    if include_fully_connected:
        x = Dense(n_output_classes, activation=activation, 
                  kernel_regularizer=l2(weight_decay), 
                  bias_regularizer=l2(weight_decay))(x)

    return x


    
    
    
    
    