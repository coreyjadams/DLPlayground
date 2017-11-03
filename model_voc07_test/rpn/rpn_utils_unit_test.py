from models import rpn_utils

# # Testing the generation of anchors and operations with them:
# # First test anchor generation:
# _base_anchors = rpn_utils.generate_anchors(base_size = 16,
#                                            ratios = [0.5, 1.0, 2.0],
#                                            scales = [1./16, 0.5, 2, 16] )

# # Based on this, there should be 12 anchors total.  4 should have L/W ratio of 0.5,
# # 4 of 1.0, 4 of 2.0.  3 should have area 1, 3 should have area 8*8, 3 should
# # have area 32*32 = 1024, and 3 should have area 16*16*16*16 = 66536

# # All anchors should be centered at base_size/2

# # Some small rounding errors in the area are acceptable, since it's rounded
# # to the nearest int

# for anchor in _base_anchors:
#     print "{}: area {}, AspectRatio {}".format(anchor, 
#                                                anchor[2]*anchor[3],
#                                                anchor[2]/anchor[3])

# # Now convert the anchors to the min max format, and check the values:
# _min_max_anchors = rpn_utils.anchors_whctrs_to_minmax(_base_anchors, in_place=False)

# for anchor in _min_max_anchors:
#     print "{}: area {}, AspectRatio {}".format(anchor, 
#                                                (anchor[2] - anchor[0])*(anchor[3] - anchor[1]),
#                                                (anchor[2] - anchor[0])/(anchor[3] - anchor[1]))

# Looks like the conversion works.  Let's do some tests of the IoU 
# by generating new anchors and padding them to known IoUs
_base_anchors = rpn_utils.generate_anchors(base_size = 16,
                                           ratios = [1.0],
                                           scales = [1.0] )

_padded_anchors = rpn_utils.pad_anchors(_base_anchors, 
                                        n_tiles_x = 3,
                                        n_tiles_y = 1,
                                        step_size_x = 8,
                                        step_size_y = 0)

print _padded_anchors
# These anchors are identical except that the x_center is shifted by half of the box
# Therefore, the intersection should be 0.5*box_area, and the union should be 
# 1.5*box_area, for an IoU of 0.333.
# The boxs' IoU with themselves should be 1.0:
# print rpn_utils.numpy_IoU_xyctrs(_padded_anchors[0], _padded_anchors[1])
print rpn_utils.numpy_IoU_xyctrs(_padded_anchors, _padded_anchors)

#we should get the same answers after converting to the other format:
_converted_anchors = rpn_utils.anchors_whctrs_to_minmax(_padded_anchors)
print _converted_anchors
print rpn_utils.numpy_IoU_minmax(_converted_anchors,
                                 _converted_anchors)
