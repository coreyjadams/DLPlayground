import tensorflow as tf
from network import network, hyperparameters


class rpn_params(hyperparameters):

    def __init__(self):
        super(rpn_params, self).__init__()

        # Parameters that are important to rpn:
        self._network_params['n_blocks'] = 18
        self._network_params['include_fully_connected'] = False
        self._network_params['n_initial_filters'] = 16
        self._network_params['downsample_interval'] = 8
        self._network_params['initial_stride'] = 1
        self._network_params['initial_kernel'] = 3
        self._network_params['bottleneck'] = False
        self._network_params['weight_decay'] = 1E-3
        self._network_params['activation'] = 'softmax'

        self._key_param_dict.update({"n_blocks": "nb",
                                     "n_initial_filters": "nf",
                                     "downsample_interval" : "di",
                                     "initial_stride": "is",
                                     "initial_kernel": "ik",
                                     "bottleneck": "B",
                                     })


class rpn(network):
    """docstring for rpn"""

    def __init__(self, params=None):
        name = "rpn"
        if params is None:
            print "Creating default params"
            params = rpn_params()
        super(rpn, self).__init__(name, params)

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
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
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
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2D",
                                 reuse=None)

        # Sum the input and the output:
        with tf.variable_scope(name+"_add"):
          x = tf.add(x, input_tensor, name="Add")
        return x



    def downsample_block(self, input_tensor,
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
        n_filters = 2*input_tensor.get_shape().as_list()[-1]

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
                                 kernel_size=[3, 3],
                                 strides=[2, 2],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
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
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2D",
                                 reuse=None)

        # Map the input tensor to the output tensor with a 1x1 convolution
        with tf.variable_scope(name+"identity"):
            y = tf.layers.conv2d(input_tensor,
                                 n_filters,
                                 kernel_size=[1, 1],
                                 strides=[2, 2],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2D1x1",
                                 reuse=None)

        # Sum the input and the output:
        with tf.variable_scope(name+"_add"):
            x = tf.add(x, y)
        return x

    def build_network(self, input_tensor, is_training=True, n_output_classes=10):

        params = self._params.network_params()

        # Initial convolutional layer:
        x = tf.layers.conv2d(input_tensor,
                             params['n_initial_filters'],
                             kernel_size=(params['initial_kernel'],
                                          params['initial_kernel']),
                             strides=(params['initial_stride'],
                                      params['initial_stride']),
                             padding='same',
                             activation=None,
                             use_bias=False,
                             bias_initializer=tf.zeros_initializer(),
                             trainable=True,
                             name="InitialConv2D",
                             reuse=None)

        for i in xrange(params["n_blocks"]):

            if i != 0 and i % params['downsample_interval'] == 0:
                x = self.downsample_block(x, name="res_block_downsample_{}".format(i),
                                          is_training=is_training)
            else:
                x = self.residual_block(x, name="res_block_{}".format(i),
                                        is_training=is_training)

        # At this point, it has been a standard network implementation.
        # Now we want to map the convolutional feature map to a set of regression
        # stages.  First, a fully connected network that maps to a 512d space
        # 
        # As mentioned in the Faster R-CNN paper, a fully connected network is 
        # simply an nxn convolution:
        
        n = x.get_shape().as_list()[1]
        # n_filters = 2*input_tensor.get_shape().as_list()[-1]

        with tf.variable_scope("RPN-FC"):
            x = tf.layers.conv2d(x,
                                 512,
                                 kernel_size=[n, n],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2DNxN",
                                 reuse=None)

        k = (n - 2)*(n - 2)

        with tf.variable_scope("RPN-reg"):
            regressor = tf.layers.conv2d(x,
                                         4*k,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=None,
                                         use_bias=False,
                                         kernel_initializer=None,  # automatically uses Xavier initializer
                                         kernel_regularizer=None,
                                         activity_regularizer=None,
                                         trainable=True,
                                         name="Conv2D1x1-reg",
                                         reuse=None)

            # Reshape the regressor into the feature map pools it was using:
            regressor = tf.reshape(regressor, (tf.shape(regressor)[0], n-2, n-2, 4))


        with tf.variable_scope("RPN-reg"):
            classifier = tf.layers.conv2d(x,
                                          2*k,
                                          kernel_size=[1, 1],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=None,
                                          use_bias=False,
                                          kernel_initializer=None,  # automatically uses Xavier initializer
                                          kernel_regularizer=None,
                                          activity_regularizer=None,
                                          trainable=True,
                                          name="Conv2D1x1-cls",
                                          reuse=None)        

            # Reshape the classifier into the feature map pools it was using:
            classifier = tf.reshape(classifier, (tf.shape(classifier)[0], n-2, n-2, 2))

            # Apply the activation:
            classifier = tf.nn.softmax(classifier, dim=-1)

        return classifier, regressor
