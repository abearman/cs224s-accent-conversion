import numpy as np

def pad_sequence(mfcc_features, max_num_frames):
		"""
		Args:
			mfcc_features: a np.ndarray array of shape (num_frames, num_mfcc_coeffs)
			max_num_frames: the maximum length to which the array should be truncated or zero-padded 

		Returns:
			padded_mfcc_features: a np.ndarray of shape (max_num_frames, num_mfcc_coeffs)
			mask: a np.ndarray of shape (max_num_frames,)
		"""
		num_frames = mfcc_features.shape[0]
		num_mfcc_coeffs = mfcc_features.shape[1]

		padded_mfcc_features = np.zeros( (max_num_frames, num_mfcc_coeffs) )
		mask = np.zeros( (max_num_frames,), dtype=bool)

		# Truncate (or fill exactly
		if num_frames >= max_num_frames:
			padded_mfcc_features = mfcc_features[0:max_num_frames,:]
			mask = np.ones((max_num_frames,), dtype=bool)  # All True's 

		# Append 0 MFCC vectors
		elif num_frames < max_num_frames:
			delta = max_num_frames - num_frames
			zeros = np.zeros((delta, num_mfcc_coeffs))
			padded_mfcc_features = np.concatenate((mfcc_features, zeros), axis=0)

			trues = np.ones((num_frames,), dtype=bool)
			falses = np.zeros((delta,), dtype=bool)
			mask = np.concatenate((trues, falses), axis=0)

		return (padded_mfcc_features, mask)
