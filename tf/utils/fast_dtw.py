import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


def get_dtw_series(source_mfccs, target_mfccs):
	"""As in the FastDTW paper, this function transforms two time series, 
	of different lengths (x_orig, y_orig) ==> (x_warped, y_warped) of the 
	same length. Here, the time series are both matrics of MFCC features.

	Args:
		source_mfccs: A matrix of source MFCCs of shape (num_frames_in_source, num_coefficients)
		target_mfccs: A matrix of target MFCCs of shape (num_frames_in_target, num_coefficients)
	Returns:
		source_warped: The warped matrix of shape (num_frames_warped, num_coefficients) 
		target_warped: The warped matrix of shape (num_frames_warped, num_coefficients)
	"""
	source_list = source_mfccs.tolist()
	target_list = target_mfccs.tolist()
	distance, path = fastdtw(source_list, target_list, dist=euclidean)
	print "distance: ", distance

	num_warped_frames = len(path)
	num_coefficients = 1
	if len(source_mfccs.shape) > 1:
		num_coefficients = source_mfccs.shape[1]
	source_warped_array = np.zeros( (num_warped_frames, num_coefficients) ) 
	target_warped_array = np.zeros( (num_warped_frames, num_coefficients) )

	index = 0  # Need this index to mimic the "append" function for lists
	for src_idx, tgt_idx in path:
		source_warped_array[index,:] = source_list[src_idx]
		target_warped_array[index,:] = target_list[tgt_idx]
		index += 1

	return np.squeeze(source_warped_array), np.squeeze(target_warped_array)
