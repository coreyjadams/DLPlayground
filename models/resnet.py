import tensorflow as tf
from network import network, hyperparameters


class resnet_params(hyperparameters):

    def __init__(self):
        super(resnet_params, self).__init__()

        # Parameters that are important to resnet:
        self._network_params['n_blocks'] = 18
        self._network_params['include_final_classifier'] = True
        self._network_params['n_classes'] = 10
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


class resnet(network):
    """docstring for resnet"""

    def __init__(self, params=None):
        name = "resnet"
        if params is None:
            print "Creating default params"
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

    def build_network(self, input_tensor, is_training=True):

        params = self._params.network_params()
        n_output_classes = params['n_classes']

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

        # A final convolution to map the features onto the right space:
        if params["include_final_classifier"]:
          with tf.variable_scope("final_pooling"):
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
              x = tf.nn.relu(x, name="final_pooling")

              x = tf.layers.conv2d(x,
                                   n_output_classes,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   data_format='channels_last',
                                   dilation_rate=(1, 1),
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=None,  # automatically uses Xavier initializer
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
              x = tf.reshape(x, [tf.shape(x)[0], n_output_classes],
                             name="global_pooling_reshape")
              # Apply the activation:
              x = tf.nn.softmax(x, dim=-1)

        return x
