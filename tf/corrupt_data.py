import numpy as np
import tensorflow as tf

# All code in this file comes from https://github.com/blackecho/Deep-Learning-TensorFlow/blob/master/yadlt/utils/utilities.py

def corrupt_input(data, corr_frac=0.5, corr_type='masking'):

		""" Corrupt a fraction 'corr_frac' of 'data' according to the
		noise method of this autoencoder.
		
		Args:
			data: The input data to corrupt, a (max_num_frames x num_coefficients) matrix
			corr_frac: The fraction of data to corrupt (a float between 0 and 1)

		Returns: corrupted data
		"""
		# Calculates how many actual features should be corrupted
		corruption_ratio = np.round(corr_frac * data.shape[1]).astype(np.int)

		if corr_type == 'masking':
				x_corrupted = masking_noise(data, corruption_ratio)

		elif corr_type == 'salt_and_pepper':
				x_corrupted = salt_and_pepper_noise(data, corruption_ratio)

		elif corr_type == 'none':
				x_corrupted = data

		else:
				x_corrupted = None

		return x_corrupted


def masking_noise(data, sess, v):
    """Apply masking noise to data in X.
    In other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param data: array_like, Input data
    :param sess: TensorFlow session
    :param corr_frac: fraction of elements to distort, float
    :return: transformed data
    """
    data_noise = data.copy()
    rand = tf.random_uniform(data.shape)
    data_noise[sess.run(tf.nn.relu(tf.sign(v- rand))).astype(np.bool)] = 0

    return data_noise


def salt_and_pepper_noise(X, v):
    """Apply salt and pepper noise to data in X.
    In other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a
    fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.
    :param X: array_like, Input data
    :param v: int, fraction of elements to distort
    :return: transformed data
    """
    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise

