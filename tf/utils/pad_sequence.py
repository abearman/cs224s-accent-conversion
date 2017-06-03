import numpy as np

def pad_sequence(features, max_num_frames, num_samples_per_frame=None):
		"""
		Args:
			features: Either a list of len num_features or a np.ndarray array of shape (num_frames, num_mfcc_coeffs)
			max_num_frames: the maximum length to which the array should be truncated or zero-padded 

		Returns:
			padded_features: a list of len max_num_frames or a np.ndarray of shape (max_num_frames, num_mfcc_coeffs)
			mask: a np.ndarray of shape (max_num_frames,)
		"""

		# features is a 1D array 
		if len(features.shape) == 1:
			num_samples = len(features)	
			max_num_samples = max_num_frames * num_samples_per_frame
			padded_features = np.zeros( (max_num_samples,) )

			# Truncate (or fill exactly)
			if num_samples >= max_num_samples: 
				padded_features = features[0:max_num_samples]
	
			# Append 0 samples
			else:
				delta = max_num_samples - num_samples 
				zeros = np.zeros((delta,))
				padded_features = np.concatenate((features, zeros), axis=0)

			return padded_features

		# features is a 2D array
		else: 
			num_frames = features.shape[0]
			num_mfcc_coeffs = features.shape[1]

			padded_features = np.zeros( (max_num_frames, num_mfcc_coeffs) )
			mask = np.zeros( (max_num_frames,), dtype=bool)

			# Truncate (or fill exactly)
			if num_frames >= max_num_frames:
				padded_features = features[0:max_num_frames,:]
				mask = np.ones((max_num_frames,), dtype=bool)  # All True's 

			# Append 0 MFCC vectors
			elif num_frames < max_num_frames:
				delta = max_num_frames - num_frames
				zeros = np.zeros((delta, num_mfcc_coeffs))
				padded_features = np.concatenate((features, zeros), axis=0)

				trues = np.ones((num_frames,), dtype=bool)
				falses = np.zeros((delta,), dtype=bool)
				mask = np.concatenate((trues, falses), axis=0)

			return (padded_features, mask)
