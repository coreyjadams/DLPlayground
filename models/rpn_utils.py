import numpy as np
import random

def boxes_whctrs_to_minmax(boxes, in_place = False):
    """
    Map boxes from the (x_ctr, y_ctr, width, height) format
    to the (x_min, y_min, x_max, y_max) format

    This should work equally well for np arrays or tf tensors
    """

    if not in_place:
        boxes = np.copy(boxes)

    # Move the centers to be the minima:
    boxes[:,0] -= 0.5*boxes[:,2]
    boxes[:,1] -= 0.5*boxes[:,3]

    # Add the width to the start to get the max:
    boxes[:,2] += boxes[:,0] 
    boxes[:,3] += boxes[:,1] 

    return boxes

def boxes_minmax_to_whctrs(boxes, in_place = False):
    """
    Map boxes from the (x_min, y_min, x_max, y_max) format
    to the (x_ctr, y_ctr, width, height) format

    This should work equally well for np arrays or tf tensors
    """
    if not in_place:
        boxes = np.copy(boxes)
    # Calculate the widths:
    boxes[:,2] = boxes[:2] - boxes[:,0]
    boxes[:,3] = boxes[:3] - boxes[:,1]

    # Move the min to the center:
    boxes[:,0] += 0.5*boxes[:,2]
    boxes[:,1] += 0.5*boxes[:,3]

    return boxes

def boxes_regcoord_to_whctrs(boxes, anchors):
    """
    Map the regressed coordinates to real coordinates
    based on the anchors
    """
    # R[0] == t_x == (x - x_A) / w_A # The difference in regressed coord / width
    # R[1] == t_y == (y - y_A) / h_A # The difference in regressed coord / height
    # R[2] == t_w == log(w/w_A) # log ratio of widths
    # R[3] == t_h == log(h/h_A) # log ratio of heights
    # 
    boxes[:,2] = np.exp(boxes[:,2]) * anchors[:,2] 
    boxes[:,3] = np.exp(boxes[:,3]) * anchors[:,3] 

    # Width and height are converted, now do coordinates:
    boxes[:,0] = anchors[:,0] + anchors[:,2] * boxes[:,0] 
    boxes[:,1] = anchors[:,1] + anchors[:,3] * boxes[:,1] 

    return boxes

def pad_anchors(input_anchors, n_tiles_x=32, n_tiles_y=32, 
                step_size_x=16, step_size_y=16):
    """
    Take a set of anchors and expand them to cover the full image.
    Anchors need to be in the (x_ctr, y_ctr, width, height) format.

    There is no check on step_size, so you can always make it negative
    to pad the list of anchors in the opposite direction.
    """
    n_input_anchors = len(input_anchors)
    
    # Map the main anchors to cover every n pixels:
    _list_of_anchors = np.tile(input_anchors, reps=(n_tiles_x,n_tiles_y,1,1))
    
    for i in xrange(n_tiles_x):
        _list_of_anchors[i,:,:,0] += step_size_x*i
    for i in xrange(n_tiles_y):
        _list_of_anchors[:,i,:,1] += step_size_y*i

    # Flatten the result into a list of anchors (final shape [n_anchors, 4])
    return np.reshape(_list_of_anchors, (n_input_anchors*n_tiles_x*n_tiles_y, 4))


def generate_anchors(base_size = 16, ratios = [0.5, 1, 2.0], 
                      scales = 2**np.arange(4,7)):
    """
    Generate anchors.
    For each ratio and scale, there will be an anchor generated
    with area of (scale*base_size)**2, centered at 
    (base_size/2, base_size/2), with aspect ratio X:Y of ratio
    
    Anchors come out unordered.
    
    """
    n_anchors = len(ratios)*len(scales)

    base_anchor = np.array([int(base_size*0.5), 
                            int(base_size*0.5), 
                            base_size, 
                            base_size], np.float32) 
    # First, generate a list of anchors of the appropriate area:
    scaled_anchors = np.tile(base_anchor,(len(scales), 1))
    final_anchors = np.zeros((len(scales),len(ratios), 4), np.float32)
    for s, i in zip(scales, xrange(len(scales))):
        scaled_anchors[i,2:] *= s
        for r, j in zip(ratios, xrange(len(ratios))):
            t = np.copy(scaled_anchors[i])
            t[2] *= np.sqrt(r)
            t[3] *= 1./np.sqrt(r)
            final_anchors[i,j] = np.round(t)
    return np.reshape(final_anchors, (n_anchors, 4))


def prune_noninternal_anchors(_anchors, image_size):

    inds_inside = np.where(
        (_anchors[:, 0] >= image_size[0]) &
        (_anchors[:, 1] >= image_size[1]) &
        (_anchors[:, 2] <  image_size[2]) &  # width
        (_anchors[:, 3] <  image_size[3])    # height
    )[0]

    return _anchors[inds_inside]


def numpy_IoU_xyctrs(bb1, bb2):
    """
    @brief      Compute the Intersection over Union for two bounding boxes
    
    @param      bb1   Bounding Box 1, as (x_center, y_center, width, height)
                    or (n1, x_center, y_center, width, height)
    @param      bb2   Bounding Box 2, as (x_center, y_center, width, height)
                    or (n2, x_center, y_center, width, height)
    
    @return     Intersection over Union value, between 0 and 1
    """

    # There should be at least one anchor, but maybe more.  
    # If the number of dimensions is only 1 (so it's just
    # one anchor), reshape the array to allow the slicing
    # to work properly:
    
    if bb1.ndim == 1:
        bb1 = np.reshape(bb1, (1,) + bb1.shape)
    if bb2.ndim == 1:
        bb2 = np.reshape(bb2, (1,) + bb2.shape)

    # Now, n1 and n2 can represent the number of anchors:
    n_1 = bb1.shape[0]
    n_2 = bb2.shape[0]

    # Want the IoU for every bb1 to every bb2, so tile them into
    # long 1D arrays to allow slicing:
    bb1_arr = np.reshape(np.tile(bb1, [1, n_2]), (n_1*n_2, 4))
    bb2_arr = np.tile(bb2, [n_1, 1])


    x1 = np.max((bb1_arr[:,0] - 0.5*bb1_arr[:,2], 
                bb2_arr[:,0] - 0.5*bb2_arr[:,2]),
                axis=0)
    y1 = np.max((bb1_arr[:,1] - 0.5*bb1_arr[:,3], 
                bb2_arr[:,1] - 0.5*bb2_arr[:,3]),
                axis=0)
    x2 = np.min((bb1_arr[:,0] + 0.5*bb1_arr[:,2], 
                bb2_arr[:,0] + 0.5*bb2_arr[:,2]),
                axis=0)
    y2 = np.min((bb1_arr[:,1] + 0.5*bb1_arr[:,3], 
                bb2_arr[:,1] + 0.5*bb2_arr[:,3]),
                axis=0)

    w = x2 - x1
    h = y2 - y1

    inter = w*h

    aarea  = (bb1_arr[:,3])* (bb1_arr[:,2])
    barea  = (bb2_arr[:,3])* (bb2_arr[:,2])

    denom = aarea + barea - inter
    mask = (denom == 0)
    denom[mask] = 0.1

    IoU = inter / (denom)
    IoU[mask] = 0
    IoU[w <= 0] = 0
    IoU[h <= 0] = 0
    
    return np.squeeze(np.reshape(IoU, (n_1,n_2)))


def numpy_IoU_minmax(bb1, bb2):
    """
    @brief      Compute the Intersection over Union for two bounding boxes
    
    @param      bb1   Bounding Box 1, as (x_min, y_min, x_max, y_max)
                    or (n1), ( x_min, y_min, x_max, y_max)
    @param      bb2   Bounding Box 2, as (x_min, y_min, x_max, y_max)
                    or (n2), ( x_min, y_min, x_max, y_max)
    
    @return     Intersection over Union value, between 0 and 1
    """

    # There should be at least one anchor, but maybe more.  
    # If the number of dimensions is only 1 (so it's just
    # one anchor), reshape the array to allow the slicing
    # to work properly:
    
    if bb1.ndim == 1:
        bb1 = np.reshape(bb1, (1,) + bb1.shape)
    if bb2.ndim == 1:
        bb2 = np.reshape(bb2, (1,) + bb2.shape)

    # Now, n1 and n2 can represent the number of anchors:
    n_1 = bb1.shape[0]
    n_2 = bb2.shape[0]


    # Want the IoU for every bb1 to every bb2, so tile them into
    # long 1D arrays to allow slicing:
    bb1_arr = np.reshape(np.tile(bb1, [1, n_2]), (n_1*n_2, 4))
    bb2_arr = np.tile(bb2, [n_1, 1])


    x1 = np.max((bb1_arr[:,0], 
                bb2_arr[:,0]),
                axis=0)
    y1 = np.max((bb1_arr[:,1], 
                bb2_arr[:,1]),
                axis=0)
    x2 = np.min((bb1_arr[:,2], 
                bb2_arr[:,2]),
                axis=0)
    y2 = np.min((bb1_arr[:,3], 
                bb2_arr[:,3]),
                axis=0)


    w = x2 - x1
    h = y2 - y1


    inter = w*h

    aarea  = (bb1_arr[:,3] - bb1_arr[:,1])*(bb1_arr[:,2] - bb1_arr[:,0])
    barea  = (bb2_arr[:,3] - bb2_arr[:,1])*(bb2_arr[:,2] - bb2_arr[:,0])


    denom = aarea + barea - inter
    # mask = (denom == 0)
    # denom[mask] = 0.1

    IoU = inter / (denom)
    # return IoU
    # IoU[mask] = 0
    IoU[w <= 0] = 0
    IoU[h <= 0] = 0
    
    return np.reshape(IoU, (n_1,n_2))

def numpy_select_label_anchors_minmax(_ground_truths, 
                                      _anchors, 
                                      _positive_threshold = 0.7, 
                                      _negative_threshold = 0.3,
                                      _n_targets= 128):
    """
    Compute the labels for anchors based on ground truth.
    Anchors that have IOU with a ground truth above _positve_threshold
    are tagged as positive, similarly if IOU is below _negative_threshold
    with all ground_truths the anchor is negative
    """
    
    # Prune anchors that sit over the edge of the images
    


    _possible_anchors = _anchors 

    # Compute the IoU for these anchors with the ground truth boxes:
    iou = numpy_IoU_minmax(_ground_truths, _possible_anchors)




    # For each ground truth, we select the anchor that has the highest overlap:
    

    _best_anchors = np.argmax(iou, axis=-1)
    _best_anchor_iou = np.max(iou, axis=1)


    gt_id = np.arange(0, len(_best_anchors))


    _best_anchor_indexs = [gt_id, _best_anchors]
        
    # Additionally, select anchors with an IOU greater than the positive threshold:
    _positive_anchors = np.where(iou > 0.7)
    
    # For negative anchors, we need to make sure each anchor has an IoU with all 
    # ground truth that is at most 0.3
    _worst_anchors = np.max(iou.T, axis=1)

    _neg_anchors = np.where(_worst_anchors < 0.3)

    # print _neg_anchors[0][0:5]
    # print _neg_anchors[1][0:5]

    # We need to keep track of the label for each anchor, as well as it's matched
    # ground truth box

    labels = np.full(iou.shape, -1)

    labels[_positive_anchors] = 1
    # Assign the negative labels first, so it doesn't clobber the best IoU if they 
    # are all really low
    labels[:,_neg_anchors] = 0

    labels[_best_anchor_indexs] = 1

    # Now, we know where the anchors are positive and negative (or don't care)
    # We create a list of anchors (positive and negative)    

    # Gather the positive labels:
    pos_gt, pos_anchors = np.where(labels == 1)
    n_pos_labels = len(pos_gt)

    neg_gt, neg_anchors = np.where(labels == 0)
    n_neg_labels = len(neg_gt)

    # Downselect:

    if n_pos_labels > (_n_targets / 2):
        pos_labels = np.random.choice(len(pos_gt), size=(int(_n_targets/2)), replace=False)
        pos_gt = pos_gt[pos_labels]
        pos_anchors = pos_anchors[pos_labels]
        n_pos_labels = len(pos_gt)
        pass
    if n_neg_labels > (_n_targets - n_pos_labels):
        neg_labels = np.random.choice(len(neg_gt), size=(_n_targets - n_pos_labels), replace=False)
        neg_gt = neg_gt[neg_labels]
        neg_anchors = neg_anchors[neg_labels]
        print neg_labels.shape

    # Join everything together:
    _gt = np.concatenate([pos_gt, neg_gt])
    _matched_anchors = np.concatenate([pos_anchors, neg_anchors])
    _labels = np.zeros((_n_targets))
    _labels[0:len(pos_gt)] = 1

    return _labels, _gt, _matched_anchors



    
    
    
    
