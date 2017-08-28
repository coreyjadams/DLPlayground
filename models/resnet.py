import tensorflow as tf
from network import network, hyperparameters

class resnet_params(hyperparameters):

    def __init__(self):
        super(resnet_params, self).__init__()

        # Parameters that are important to resnet:
        

class resnet(network):
    """docstring for resnet"""

    def __init__(self, params = None):
        name = "resnet"
        if params is None:
            params = resnet_params()
        super(resnet, self).__init__(name, params)


    def residual_block(self, input_tensor, 
                       is_training,
                       kernel=[3, 3], 
                       stride=[1, 1], 
                       name=""):
        """
        @brief      Create a residual block and apply it to the input tensor

        @param      self          The object
        @param      input_tensor  The input tensor
        @param      kernel        Size of convolutional kernel to apply
        @param      n_filters     Number of output filters

        @return     { Tensor with the residual network applied }
        """

        # Residual block has the identity path summed with the output of
        # BN/Relu/Conv2d applied twice



        # Assuming channels last here:
        n_filters = input_tensor.shape[-1]

        with tf.variable_scope(name + "_0"):
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
                                              name="BatchNorm",
                                              reuse=None)
            # ReLU:
            x = tf.nn.relu(x)


            # Conv2d:
            x = tf.layers.conv2d(x, n_filters,
                                 kernel_size=[3,3],
                                 strides=[1,1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None, # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2D",
                                 reuse=None)

        # Apply everything a second time:
        with tf.variable_scope(name + "_1"):
        
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
                                              name="BatchNorm",
                                              reuse=None)
            # ReLU:
            x = tf.nn.relu(x)

            # Conv2d:
            x = tf.layers.conv2d(x,
                                 n_filters,
                                 kernel_size=[3,3],
                                 strides=[1,1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None, # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2D",
                                 reuse=None)

        # Sum the input and the output:
        x = tf.add(x, input_tensor)
        return x


    def build_network(self, input_tensor, is_training=True, n_output_classes = 10):

        
        # Initial 7x7 convolutional layer:
        
            x = tf.layers.conv2d(input_tensor,
                                 64,
                                 kernel_size=(7,7),
                                 strides=(1, 1),
                                 padding='same',
                                 dilation_rate=(1, 1),
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None, # automatically uses Xavier initializer
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="InitialConv2D",
                                 reuse=None)


            for i in xrange(3):

                x = self.residual_block(x, name="res_block_{}".format(i), 
                                        is_training=is_training)

            # A final convolution to map the features onto the right space:
            with tf.variable_scope("final_pooling"):
                # Batch normalization is applied first:
                x = tf.layers.batch_normalization(x,
                                  axis = -1,
                                  momentum = 0.99,
                                  epsilon = 0.001,
                                  center = True,
                                  scale = True,
                                  beta_initializer = tf.zeros_initializer(),
                                  gamma_initializer = tf.ones_initializer(),
                                  moving_mean_initializer = tf.zeros_initializer(),
                                  moving_variance_initializer = tf.ones_initializer(),
                                  beta_regularizer = None,
                                  gamma_regularizer = None,
                                  training = is_training,
                                  trainable = True,
                                  name="BatchNorm",
                                  reuse = None)


                # ReLU:
                x = tf.nn.relu(x, name="final_pooling")
        



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
                                     name="Conv2DBottleNeck",
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
                               name="GlobalAveragePool",
                               data_format=None)


                # Reshape to remove empty dimensions:
                x = tf.reshape(x, [tf.shape(x)[0], n_output_classes], name = "global_pooling_reshape")
                # Apply the activation:
                x = tf.nn.softmax(x,dim=-1)

            return x