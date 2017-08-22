import tensorflow as tf

# Implementation of densenet using purely tensorflow operations
# Using the functional implementation as much as possible, not the lower level implementation

class densenet():
    
    #def __init__(self, imgs, weights=None, sess=None):
    def __init__(self):
        pass
        #self.imgs = imgs
        #self.convlayers()
        #self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)
        #if weights is not None and sess is not None:
        #    self.load_weights(weights, sess)


    def build_dense_net(self, input_tensor, n_output_classes, n_blocks = 3, n_layers_per_block = 4, 
                        include_fully_connected = False, growth_rate=16, is_training=True,
                        n_initial_filters=-1, initial_stride=3, bottleneck=False, compression_factor=1.0,
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
            is_training - true or false.  affects the computation of dropout and batch normalization
            n_initial_filters - number of initial filters to produce before the first dense block
                                (Defaults to growth_rate * 2)
            bottleneck - if True, bottleneck convolutions are applied in the 
                         convolutional blocks to reduce the number of features
            compression_factor - reduce the number of channels (features) in the transition layer with 1x1 convolution, 
                                 to reduce the input to the next denseblock.  Output of each dense blocks is 
                                 n_initial_filters + n_layers_per_block*growth_rate.  0.5 is used in the paper.
            dropout_rate - the rate at which dropout is applied.  default is none
            weight_decay - regularization parameter for weight decay in the lose function, default is 1e-4
            activation - activation function for the fully connected layer, if used.
        '''

        
        # Do some basic parameter checking:
        assert compression_factor > 0 and compression_factor <=1, 'Compression must be between 0 and 1.'

        if n_initial_filters == -1:
            n_initial_filters = 2*growth_rate

        assert n_initial_filters > 0, 'Number of initial filters must be positive.'


        # Now, start building the model
        # Initial 7x7 convolutional layer:
        x = tf.layers.conv2d(input_tensor,
                             n_initial_filters,
                             kernel_size=(7,7),
                             strides=(initial_stride, initial_stride),
                             padding='same',
                             data_format='channels_last',
                             dilation_rate=(1, 1),
                             activation=None,
                             use_bias=False,
                             kernel_initializer=None, # automatically uses Xavier initializer
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             trainable=True,
                             name="densenet",
                             # name="initial_convolution7x7",
                             reuse=None)


        # Apply dense blocks and transition blocks alternately, except no transition block after the last dense block:
        for block in xrange(n_blocks - 1):

            x = self.dense_block(x, n_layers_per_block, growth_rate, name=block,
                                 is_training=is_training,
                                 bottleneck=bottleneck, 
                                 dropout=dropout_rate, 
                                 weight_decay=weight_decay)


            x = self.transition_block(x, 4*growth_rate, compression_factor, 
                                      name=block, dropout=dropout_rate, 
                                      is_training=is_training,
                                      weight_decay=weight_decay)



        # The last dense_block:
        x = self.dense_block(x, n_layers_per_block, growth_rate, name=n_blocks-1, 
                             is_training=is_training,
                             bottleneck=bottleneck,
                             dropout=dropout_rate, weight_decay=weight_decay)

        with tf.name_scope("final_pooling"):

            # Batch normalization is applied first:
            x = tf.layers.batch_normalization(x,
                                  axis=-1,
                                  momentum=0.99,
                                  epsilon=0.001,
                                  center=True,
                                  scale=True,
                                  beta_initializer=tf.zeros_initializer(),
                                  gamma_initializer=tf.ones_initializer(),
                                  moving_mean_initializer=tf.zeros_initializer(),
                                  moving_variance_initializer=tf.ones_initializer(),
                                  beta_regularizer=None,
                                  gamma_regularizer=None,
                                  training=is_training,
                                  trainable=True,
                                  name="final_pooling",
                                  reuse=None)

            # ReLU:
            x = tf.nn.relu(x, name="final_pooling")

        # if using a fully connected layer, map this output to a fully connected layer.

        # if not using a fully connected layer, use a 1x1 convolution to map features to the right number
        # of features to apply global average pooling


        if include_fully_connected:
            pass
#                x = Dense(n_output_classes, activation=activation, 
#                          kernel_regularizer=l2(weight_decay), 
#                          bias_regularizer=l2(weight_decay))(x)

        else:
            with tf.name_scope("global_average_pooling"):
                x = tf.layers.conv2d(x,
                                     n_output_classes,
                                     kernel_size=(1,1),
                                     strides=(1, 1),
                                     padding='same',
                                     data_format='channels_last',
                                     dilation_rate=(1, 1),
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=None, # automatically uses Xavier initializer
                                     bias_initializer=tf.zeros_initializer(),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     trainable=True,
                                     name="global_average_pooling",
                                     # name="convolution_globalpool_bottleneck1x1",
                                     reuse=None)
                # For global average pooling, need to get the shape of the input:
                shape = (x.shape[1], x.shape[2])
                x = tf.nn.pool(x,
                               window_shape=shape,
                               pooling_type="AVG",
                               padding="VALID",
                               dilation_rate=None,
                               strides=None,
                               name="global_average_pooling",
                               data_format=None)


                # Reshape to remove empty dimensions:
                x = tf.reshape(x, [tf.shape(x)[0], n_output_classes], name = "global_pooling_reshape")
                # Apply the activation:
                x = tf.nn.softmax(x,dim=-1)


        return x

    def dense_block(self, x, n_layers, growth_rate, name="", is_training=True,
                    bottleneck=False, dropout=None, weight_decay=1e-4):
        '''
        x is the input tensor, LxWxK
        n_layers is the number of convolutional layers to apply
        growth_rate is the number of filters that get added at each convolutional block
        bottleneck is whether to apply bottleneck at each convolutional layer
        dropout is the dropout rate
        weight_decay is the rate of weight decay
        '''


        with tf.name_scope("denseblock_{}".format(name)):

            # For each convolutional block, concatenate it's output with it's input
            for layer in range(n_layers):
                lab =  "{}_{}".format(name, layer)
                conv_output = self.convolution_block(x, growth_rate, lab, is_training,
                                                     bottleneck, dropout, weight_decay)

                x = tf.concat([x, conv_output], axis = -1, name='denseblock_{}_concat_{}'.format(name, layer))

            return x
    
    def convolution_block(self, input_tensor, n_filters, name="", is_training=True,
                          bottleneck=False, dropout=None, weight_decay=1e-4):
        '''
        input_tensor is the input keras tensor
        n_filters is the number of filters produced at each layer of a dense block
        bottleneck determines if feature reduction is applied before each 3x3 conv
        dropout is the dropout rate
        weight_decay is the regularization term (L2)
        '''
        _scope = "conv_{}".format(name)
        with tf.name_scope(_scope):

            with tf.name_scope("conv_batch_normalization_{}".format(name)):
                # Batch normalization is applied first:
                x = tf.layers.batch_normalization(input_tensor,
                                      axis=-1,
                                      momentum=0.99,
                                      epsilon=0.001,
                                      center=True,
                                      scale=True,
                                      beta_initializer=tf.zeros_initializer(),
                                      gamma_initializer=tf.ones_initializer(),
                                      moving_mean_initializer=tf.zeros_initializer(),
                                      moving_variance_initializer=tf.ones_initializer(),
                                      beta_regularizer=None,
                                      gamma_regularizer=None,
                                      training=is_training,
                                      trainable=True,
                                      name="conv_batch_normalization_{}".format(name),
                                      reuse=None)

            # ReLU:
            x = tf.nn.relu(x, name="{}_relu_1".format(_scope))


            if bottleneck:
                # 1x1 convolution to reduce the number of filters
                with tf.name_scope("bottleneck"):
                    x = tf.layers.conv2d(x,
                                 4*n_filters,
                                 kernel_size=(1,1),
                                 strides=(1, 1),
                                 padding='same',
                                 data_format='channels_last',
                                 dilation_rate=(1, 1),
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None, # automatically uses Xavier initializer
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 # name="{}_bottleneck1x1".format(_scope),
                                 reuse=None)


                    # Dropout
                    if dropout is not None:
                        x = tf.layers.dropout(x,
                                              rate=dropout,
                                              noise_shape=None,
                                              seed=None,
                                              training=is_training,
                                              name=_scope)
                                              # name="{}_dropout_1".format(_scope))

                    # Batch Norm
                    x = tf.layers.batch_normalization(input_tensor,
                                                      axis=-1,
                                                      momentum=0.99,
                                                      epsilon=0.001,
                                                      center=True,
                                                      scale=True,
                                                      beta_initializer=tf.zeros_initializer(),
                                                      gamma_initializer=tf.ones_initializer(),
                                                      moving_mean_initializer=tf.zeros_initializer(),
                                                      moving_variance_initializer=tf.ones_initializer(),
                                                      beta_regularizer=None,
                                                      gamma_regularizer=None,
                                                      training=is_training,
                                                      trainable=True,
                                                      # name="{}_bn_2".format(_scope),
                                                      reuse=None)
                    # ReLU:
                    x = tf.nn.relu(x, name="{}_relu_2".format(_scope))


            # Apply the 3x3 convolutional layer:
            with tf.name_scope("conv3x3"):
                x = tf.layers.conv2d(x,
                                     n_filters,
                                     kernel_size=(3,3),
                                     strides=(1, 1),
                                     padding='same',
                                     data_format='channels_last',
                                     dilation_rate=(1, 1),
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=None, # automatically uses Xavier initializer
                                     bias_initializer=tf.zeros_initializer(),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     activity_regularizer=None,
                                     trainable=True,
                                     # name="{}_3x3".format(_scope),
                                     reuse=None)

            
                #Dropout
                if dropout is not None:
                    # Apply the dropout rate:
                    x = tf.layers.dropout(x,
                                          rate=dropout,
                                          noise_shape=None,
                                          seed=None,
                                          training=is_training)
                                          # name="{}_dropout_2".format(_scope))


            # Finished a convolutional block
            return x

            
            
    def transition_block(self, input_tensor, n_filters, 
                         compression_factor, 
                         name="",
                         dropout = None,
                         is_training=True,
                         weight_decay=1e-4):
        '''
        input_tensor is the input keras tensor
        n_filters is the number of filters per stage
        compression_factor is the rate at which to reduce the number of filters, using 1x1 conv
        weight_decay is the weight_decay parameter using L2 normalization
        '''
        _scope = "transition_{}".format(name)
        with tf.name_scope(_scope):

            # Batch normalization is applied first:
            x = tf.layers.batch_normalization(input_tensor,
                                              axis=-1,
                                              momentum=0.99,
                                              epsilon=0.001,
                                              center=True,
                                              scale=True,
                                              beta_initializer=tf.zeros_initializer(),
                                              gamma_initializer=tf.ones_initializer(),
                                              moving_mean_initializer=tf.zeros_initializer(),
                                              moving_variance_initializer=tf.ones_initializer(),
                                              beta_regularizer=None,
                                              gamma_regularizer=None,
                                              training=is_training,
                                              trainable=True,
                                              name=_scope,
                                              # name="transition_bn_{}_1".format(name),
                                              reuse=None)



            # ReLU:
            x = tf.nn.relu(x, name="transition_{}_relu_1".format(name))

            # 1x1 convolution to reduce the number of filters
            n_output_filters = int(n_filters*compression_factor)

            x = tf.layers.conv2d(x,
                                 n_output_filters,
                                 kernel_size=(1,1),
                                 strides=(1, 1),
                                 padding='same',
                                 data_format='channels_last',
                                 dilation_rate=(1, 1),
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None, # automatically uses Xavier initializer
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name=_scope,
                                 # name="transition_{}_conv1x1".format(name),
                                 reuse=None)



            if dropout is not None:
                # Apply the dropout rate:
                x = tf.layers.dropout(x,
                                      rate=dropout,
                                      noise_shape=None,
                                      seed=None,
                                      training=is_training,
                                      name=_scope)
                                      # name="transition_{}_dropout".format(name))


            # Average pooling 2D:
            x = tf.layers.average_pooling2d(x,
                                            pool_size=(2,2),
                                            strides=(2,2),
                                            padding='same',
                                            data_format='channels_last',
                                            name=_scope)
                                            # name="transition_{}_avpool".format(name))

            return x
            
            
            
            