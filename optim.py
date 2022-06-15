# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import low_pass_filter as lpf
# from scipy.spatial import distance
# from scipy.optimize import minimize

# def cost(params, alpha):
#     split_coords = np.array_split(params, 2)
#     new_bp1 = np.array_split(split_coords[0], 554)
#     new_bp2 = np.array_split(split_coords[1], 554)
#     new_length = np.zeros((new_bp1.shape[0], 1))
#
#     for i in range(new_bp1.shape[0]):
#         new_length[i] = distance.euclidean(new_bp1[i], new_bp2[i])
#     length_cost = np.sum(abs(new_length - orig_skel_length))
#
#     new_dist = np.zeros((orig_wrist.shape[0], 1))
#     for i in range(orig_wrist.shape[0]):
#         new_dist[i] = distance.euclidean(new_wrist[i], np.array(orig_wrist.iloc[i])) + \
#                       distance.euclidean(new_thumb[i], np.array(orig_thumb.iloc[i]))
#     dist_cost = np.sum(new_dist)
#     c = (alpha * length_cost) + ((1 - alpha) * dist_cost)
#     return c
