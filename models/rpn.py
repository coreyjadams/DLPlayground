import tensorflow as tf
from network import network, hyperparameters
import numpy as np

class rpn_params(hyperparameters):

    def __init__(self):
        super(rpn_params, self).__init__()

        # Parameters that are important to rpn:
        self._network_params['n_anchors_per_box'] = 9
        self._network_params['weight_decay'] = 1E-3

        self._key_param_dict.update({})


class rpn(network):
    """docstring for rpn"""

    def __init__(self, params=None):
        name = "rpn"
        if params is None:
            print "Creating default params"
            params = rpn_params()
        super(rpn, self).__init__(name, params)

    def build_rpn(self, final_conv_layer, is_training=True):

        params = self._params.network_params()

        # We want to map the convolutional feature map to a set of regression
        # stages.  First, a fully connected network that maps to a 512d space
        # 
        # As mentioned in the Faster R-CNN paper, a fully connected network is 
        # simply an nxn convolution:
        
        n = final_conv_layer.get_shape().as_list()[1]
        m = final_conv_layer.get_shape().as_list()[2]
    
        with tf.variable_scope("RPN-FC"):
            x = tf.layers.conv2d(final_conv_layer,
                                 512,
                                 kernel_size=[n, m],
                                 strides=[1, 1],
                                 padding='valid',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=True,
                                 name="Conv2DNxN",
                                 reuse=None)


        k = (n - 2)*(m - 2)*params['n_anchors_per_box']

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
            regressor = tf.reshape(regressor, (tf.shape(regressor)[0], k,4))


        with tf.variable_scope("RPN-cls"):
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
            classifier = tf.reshape(classifier, (tf.shape(classifier)[0], k, 2))

            # Apply the activation:
            classifier = tf.nn.softmax(classifier, dim=-1)

        return classifier, regressor
  
    def downselect(self, regressor, classifier, anchors, ground_truth):
        """
        @brief      downselect regression output
        
        @param      self          The object
        @param      regressor     The regressor
        @param      classifier    The classifier
        @param      anchors       The anchors
        @param      ground_truth  The ground truth
        
        @return     { description_of_the_return_value }
        
        This function explicitly assumes batch is not an index in tensors
        That is, only one batch at a time.

        """

        params = self._params.network_params()


        #Anchors and ground_truth are in min/max format

        # To select anchors and compute the lost, we have to convert
        # the regression output into real image coordinates.
        # 
        # Each regressor output R has a matched anchor, A.  The 4 components
        # of R are related to A:
        # R[0] == t_x == (x - x_A) / w_A # The difference in regressed coord / width
        # R[1] == t_y == (y - y_A) / h_A # The difference in regressed coord / height
        # R[2] == t_w == log(w/w_A) # log ratio of widths
        # R[3] == t_h == log(h/h_A) # log ratio of heights

        

        with tf.variable_scope("rpn_reg_to_anchor"):
            reg_width = tf.exp(regressor[:,2]) * (anchors[:,2] - anchors[:,0])
            reg_height = tf.exp(regressor[:,3]) * (anchors[:,3] - anchors[:,1])
            # reg_boxes[:,:,2] = tf.exp(reg_boxes[:,:,2]) # * (anchors[:,2] - anchors[:,0])
            # reg_boxes[:,:,3] = tf.exp(reg_boxes[:,:,3]) * (anchors[:,3] - anchors[:,1])

            # Width and height are converted, now do coordinates:
            reg_x_ctr = regressor[:,0] * (anchors[:,2] - anchors[:,0]) + 0.5*(anchors[:,0] + anchors[:,2])
            reg_y_ctr = regressor[:,1] * (anchors[:,3] - anchors[:,1]) + 0.5*(anchors[:,1] + anchors[:,3])

            # Now convert from ctr/wh to min/max:
            
            # Move the centers to be the minima:
            reg_x_min = reg_x_ctr - 0.5*reg_width
            reg_y_min = reg_y_ctr - 0.5*reg_height

            # Add the width to the start to get the max:
            reg_x_max = reg_x_min + reg_width
            reg_y_max = reg_y_min + reg_height

        # Prune regions that extend outside the image limits:
        
        with tf.variable_scope("rpn_roi_pruning"):
            _good_x_min = tf.transpose(tf.where(reg_x_min > 0))
            _good_y_min = tf.transpose(tf.where(reg_y_min > 0))
            _good_x_max = tf.transpose(tf.where(reg_x_max < 512))
            _good_y_max = tf.transpose(tf.where(reg_y_max < 512))


            # Merge these all together
            _good_min = tf.sets.set_intersection(_good_x_min, _good_y_min)
            _good_max = tf.sets.set_intersection(_good_x_max, _good_y_max)
            _good_indexes = tf.sets.set_intersection(_good_min, _good_max)

            _good_indexes = tf.sparse_to_dense(_good_indexes.indices,
                                               _good_indexes.dense_shape,
                                               _good_indexes.values)


            # Prune the bad elements:
            reg_x_min = tf.squeeze(tf.transpose(tf.gather(reg_x_min, _good_indexes)))
            reg_x_max = tf.squeeze(tf.transpose(tf.gather(reg_x_max, _good_indexes)))
            reg_y_min = tf.squeeze(tf.transpose(tf.gather(reg_y_min, _good_indexes)))
            reg_y_max = tf.squeeze(tf.transpose(tf.gather(reg_y_max, _good_indexes)))

            # Also prune the anchors to match:
            anchors = tf.gather(anchors, _good_indexes)

            # return reg_x_min,reg_y_max,reg_x_min,reg_x_min,

        # Now, reg_x/y_min/max represents the regression output in min/max form
        # We can next compute the IoU to the ground truth boxes:
        
        with tf.variable_scope("rpn_roi_selection"):

            true_x_min, true_y_min, true_x_max, true_y_max = tf.split(ground_truth, 4, axis=-1)

            reg_coords = tf.stack([reg_x_min, reg_y_min, reg_x_max, reg_y_max], axis = 1)



            x1 = tf.maximum(true_x_min, reg_x_min)
            y1 = tf.maximum(true_y_min, reg_y_min)
            x2 = tf.minimum(true_x_max, reg_x_max)
            y2 = tf.minimum(true_y_max, reg_y_max)


            w = x2 - x1
            h = y2 - y1

            w = tf.nn.relu(w)
            h = tf.nn.relu(h)

            inter = (w) * (h )

            area1 = (reg_x_max  - reg_x_min)  * (reg_y_max  - reg_y_min)
            area2 = (true_x_min - true_x_max) * (true_y_min - true_y_max)
            denom = area1 + area2 - inter

            iou = inter / (area1 + area2 - inter)
            iou = tf.nn.relu(iou)

            # return iou, reg_coords, iou, iou

            # At this point, the iou is calculated and has the format
            # [batch][truth_box][regressed_box]

            # What we want is a selection of regressed boxes with score > 0.7
            # and non max suppression.
            # 
            # First, select the boxes where the score is greater than 0.7:
            
            # For each regression box, select the truth box it matches to
            # most closely:

            # Best score for each regression box:
            scores = tf.squeeze(tf.reduce_max(iou, axis=0))
            scores_index = tf.squeeze(tf.argmax(iou, axis=0))

            # Keep all of the indexes where the IoU is better than 0.7:
            pos_anchor_indexes = tf.where(scores > 0.7)

            # One issue with non_max_suppression: if no ROI is above threshold,
            # Take the one with the highest score:
            top_score_index = tf.argmax(scores)
            # The top score index has to be reshaped to allow gather to work
            # properly later on:
            top_score_index = tf.expand_dims(top_score_index,axis=0)

            # Now, use a conditional to determine if the pos_box_index should be
            # the top_score_index alone or the output of non_max_suppression:

            ious_above_threshold = tf.count_nonzero(pos_anchor_indexes) > 0

            pos_anchor_indexes = tf.cond(ious_above_threshold,
                                        true_fn = lambda: pos_anchor_indexes,
                                        false_fn = lambda: top_score_index)


            # Select all of the boxes where the value was better than 0.7 (or is top score):
            # Only squeeze the middle axis to keep the result 2D, even if there
            # is only one positive anchor
            pos_reg_coords = tf.gather(reg_coords, pos_anchor_indexes)
            
            
            # And their score:
            # (The score also has to be wrapped in a conditional, to squeeze it 
            # only when there is more than one score:)
            pos_scores = tf.gather(scores, pos_anchor_indexes)
            pos_scores = tf.cond(tf.rank(pos_scores) > 1,
                true_fn = lambda: tf.squeeze(pos_scores, 1),
                false_fn = lambda: pos_scores)

            pos_reg_coords = tf.cond(tf.rank(pos_reg_coords) > 2,
                true_fn = lambda: tf.squeeze(tf.gather(reg_coords, pos_anchor_indexes), axis=1),
                false_fn = lambda: pos_reg_coords)


        with tf.variable_scope("rpn_non_max_suppression"): 
            # # Non max suppress the boxes where the score was > 0.7
            # # ####TODO
            # # I think maybe this should use the classifier score?  Not sure.
            # # Currently using IoU w/ ground truth score
            pos_box_index = tf.image.non_max_suppression(
                                        boxes=pos_reg_coords, 
                                        scores=pos_scores, 
                                        max_output_size=params['n_selected_regressors']/2, 
                                        iou_threshold=0.7)

            # return pos_box_index, pos_anchor_indexes, 

            # Use the output of the non max suppression to slice out the 
            # passed indexes that correspond to the original list of regressors
            pos_anchor_indexes = tf.gather(pos_anchor_indexes, pos_box_index)

            #Then, use the final list of anchor indexes to slice out the 
            #relevant ground truth boxes:
            pos_truth_indexes = tf.gather(scores_index, pos_anchor_indexes)

        with tf.variable_scope("rpm_neg_roi_selection"):
            # Ok, do the same for the negative examples:

            # Now, gather the boxes that had a max IoU of 0.3 with all truth boxes:
            neg_anchor_indexes = tf.where(scores < 0.3)
            # Select all of the boxes where the value was better than 0.7:
            neg_reg_coords = tf.squeeze(tf.gather(reg_coords, neg_anchor_indexes ), axis=1)
            # And their score:
            neg_scores = tf.squeeze(tf.gather(scores, neg_anchor_indexes))


            # Do non max suppression on these entries too:
            neg_box_index = tf.image.non_max_suppression(
                boxes=neg_reg_coords, 
                scores=neg_scores, 
                max_output_size=params['n_selected_regressors'], 
                iou_threshold=0.7)
            


            # Use the output of the non max suppression to slice out the 
            # passed indexes that correspond to the original list of regressors
            neg_anchor_indexes = tf.gather(neg_anchor_indexes, neg_box_index)


            neg_truth_indexes = tf.gather(scores_index, neg_anchor_indexes)


        return (pos_anchor_indexes, 
                pos_truth_indexes, 
                neg_anchor_indexes, 
                neg_truth_indexes)

        # return selected_regressors, selected_classifiers, values

    def regression_loss(self, regressor, pos_box_inds, 
                        pos_true_inds, ground_truth, anchors):
        """
        @brief      Compute regression loss for RPN network
        
        @param      regressor       The full output of the regressor
        @param      pos_box_inds    The regressor boxs selected above
        @param      pos_true_inds   The corresponding true boxes
        @param      regressor       The regressor
        @param      box_label       The box label
        
        @return     { description_of_the_return_value }
        """

        # First, for all of the positive regression boxes, compute the modified
        # coordinates for the ground truth boxes to the selected anchors
        

        with tf.variable_scope("rpn_rgs_loss"):
            matched_anchors = tf.gather(anchors, pos_box_inds)

            #Similarly, gather the matching true boxes:
            matched_truths = tf.gather(ground_truth, pos_true_inds)

            #As ever, deal with the inconsistent tensor rank when there 
            #is only one successful pos_box_ind:

            matched_truths = tf.cond(tf.rank(matched_truths) > 2,
                true_fn = lambda: tf.squeeze(matched_truths, axis=1),
                false_fn = lambda: matched_truths)

            matched_anchors = tf.cond(tf.rank(matched_anchors) > 2,
                true_fn = lambda: tf.squeeze(matched_anchors, axis=1),
                false_fn = lambda: matched_anchors)

            #  Convert the truth/anchor pairs into the regression coordinates

            # R[0] == t_x == (x_true - x_A) / w_A # The difference in regressed coord / width
            # R[1] == t_y == (y_true - y_A) / h_A # The difference in regressed coord / height
            # R[2] == t_w == log(w_true/w_A) # log ratio of widths
            # R[3] == t_h == log(h_true/h_A) # log ratio of heights

            truth_reg_coord_tw = tf.log(
                (matched_truths[:,2] - matched_truths[:,0]) / (
                    matched_anchors[:,2] - matched_anchors[:,0]))
            truth_reg_coord_th = tf.log(
                (matched_truths[:,3] - matched_truths[:,1]) / 
                (matched_anchors[:,3] - matched_anchors[:,1]))
            truth_reg_coord_tx = (0.5*(matched_truths[:,2] + matched_truths[:,0]) - 
                                  0.5*(matched_anchors[:,2] + matched_anchors[:,0])
                                  ) / (matched_anchors[:,2] - matched_anchors[:,0])
            truth_reg_coord_ty = (0.5*(matched_truths[:,3] + matched_truths[:,1]) - 
                                  0.5*(matched_anchors[:,3] + matched_anchors[:,1])
                                  ) / (matched_anchors[:,3] - matched_anchors[:,1])

            truth_reg_coords = tf.stack([truth_reg_coord_tw, 
                                         truth_reg_coord_th, 
                                         truth_reg_coord_tx, 
                                         truth_reg_coord_ty], axis = 1)

            # Compute the loss now: 
            
            # Gather the selected regressors:
            positive_regressors = tf.gather(regressor, pos_box_inds)
            positive_regressors = tf.cond(tf.rank(positive_regressors) > 2,
                true_fn = lambda: tf.squeeze(positive_regressors, axis=1),
                false_fn = lambda: positive_regressors)

            loss = positive_regressors - truth_reg_coords

            # First half of the loss function: all points where 
            # |x| > 1.0
            # loss = |x| - 0.5
            loss_part1 = tf.nn.relu(tf.abs(loss) - 1) + 0.5
            # The second half of the loss function applies to |x| <= 1.0
            loss_part2 = 0.5*tf.square(loss)
            
            cond = tf.greater(loss_part2,0.5)        
            loss_part2 =  tf.where(cond,tf.zeros(tf.shape(loss)),loss_part2)

            regression_loss = tf.reduce_sum(loss_part1 + loss_part2)

            return regression_loss
            
    def classification_loss(self, classifier, pos_box_ind, neg_box_ind):
        """
        @brief      Compute classification loss for RPN
        
        @param      self         The object
        @param      classifier   The classifier
        @param      pos_box_ind  The position box ind
        @param      neg_box_ind  The negative box ind
        
        @return     Calculated loss
        """
        
        # Gather up the classifier values at the negative and 
        # positive indexes:
        with tf.variable_scope("rpn_cls_loss"):
            pos_class = tf.gather(classifier, pos_box_ind)

            pos_class = tf.cond(tf.rank(pos_class) > 2,
                true_fn = lambda: tf.squeeze(pos_class, axis=1),
                false_fn = lambda: pos_class)

            neg_class = tf.gather(classifier, neg_box_ind)

            neg_class = tf.cond(tf.rank(neg_class) > 2,
                true_fn = lambda: tf.squeeze(neg_class, axis=1),
                false_fn = lambda: neg_class)

            # Set up the "true" answers:
            pos_true = tf.zeros(tf.shape(pos_class)) + (0,1,)
            neg_true = tf.zeros(tf.shape(neg_class)) + (1,0,)

            # Now, collect pos and negative into one:
            true_labels = tf.concat((pos_true, neg_true), axis=0)
            class_labels = tf.concat((pos_class, neg_class), axis=0)

            # Finally, convert this into cross entropy loss:
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=true_labels,
                                                        logits=class_labels))

            return cross_entropy

