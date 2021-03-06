from time import time, gmtime, strftime
import numpy as np
import os
import random

import scipy.io.wavfile as wav
from python_speech_features import mfcc

import tensorflow as tf
from tensorflow.python.ops.nn import dynamic_rnn

from utils.general_utils import get_minibatches, batch_multiply_by_matrix 
from utils.fast_dtw import get_dtw_series 


class Config(object):
		"""Holds model hyperparams and data information.

		The config class is used to store various hyperparameters and dataset
		information parameters. Model objects are passed a Config() object at
		instantiation.
		"""
		batch_size = 10
		n_epochs = 1000
		lr = 1e-4
		momentum = 0.3

		sample_rate = 16000.0

		wav_len = 64000

		state_size_1 = 5000 
		state_size_2 = 5000

		dropout_keep_prob = 1.0 # 0.8
		logs_path = "tensorboard/wav_to_wav/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


class Wav2Wav(object):
		"""Implements a stacked, denoising autoencoder with a MSE loss."""

		def add_placeholders(self):
				"""Generates placeholder variables to represent the input tensors.

				These placeholders are used as inputs by the rest of the model building
				and will be fed data during training.

				Adds following nodes to the computational graph

				input_placeholder: Input placeholder tensor of shape
													 (batch_size, max_num_frames, num_mfcc_coeffs), type tf.float32
				labels_placeholder: Labels placeholder tensor of shape
														(batch_size, max_num_frames, num_mfcc_coeffs), type tf.float32
				input_masks_placeholder: Input masks placeholder tensor of shape
																 (batch_size, max_num_frames), type tf.bool
				labels_masks_placeholder: Labels masks placeholder tensor of shape
																	(batch_size, max_num_frames), type tf.bool

				Add these placeholders to self as the instance variables
						self.input_placeholder
						self.labels_placeholder
						self.input_masks_placeholder
						self.label_masks_placeholder
				"""
				self.input_placeholder = tf.placeholder(tf.float32, (None, self.config.wav_len))
				self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.wav_len))


		def create_feed_dict(self, inputs_batch, labels_batch=None):
				"""Creates the feed_dict for training the given step.

				A feed_dict takes the form of:
				feed_dict = {
								<placeholder>: <tensor of values to be passed for placeholder>,
								....
				}
				
				Args:
						inputs_batch: A batch of input data.
						input_masks_batch: A batch of input masks.
						labels_batch: (Optional) a batch of label data.
						labels_mask_batch: (Optional) a batch of label masks.
				Returns:
						feed_dict: The feed dictionary mapping from placeholders to values.
				"""
				feed_dict = {
					self.input_placeholder: inputs_batch,
					self.labels_placeholder: labels_batch
				}
				return feed_dict


		def add_prediction_op(self): 
				"""Adds the core transformation for this model which transforms a batch of input
				data into a batch of predictions. In this case, the transformation is a linear layer plus a
				softmax transformation:
			
				Implements a stacked, denoising autoencoder. 
				Autoencoder learns two mappings: (1) Encoder: input ==> hidden layer, and (2) Decoder: hidden layer ==> output layer
				

				Returns:
						pred: A tensor of shape (batch_size, n_classes)
				"""

				xavier = tf.contrib.layers.xavier_initializer()
				W1 = tf.get_variable("W1", shape=(self.config.wav_len, self.config.state_size_1), initializer=xavier) 
				b1 = tf.get_variable("b1", shape=(1, self.config.state_size_1))
				W2 = tf.get_variable("W2", shape=(self.config.state_size_1, self.config.state_size_2), initializer=xavier) 
				b2 = tf.get_variable("b2", shape=(1, self.config.state_size_2))
				W3 = tf.get_variable("W3", shape=(self.config.state_size_2, self.config.wav_len), initializer=xavier) 
			
				h1 = tf.tanh(tf.matmul(self.input_placeholder, W1))

				h2 = tf.tanh(tf.matmul(h1, W2))
				
				preds = tf.matmul(h2, W3) 

				self.predicted = preds
				return preds 
 

		def add_loss_op(self, pred):
				"""Adds mean squared error ops to the computational graph.

				Args:
						pred: A tensor of shape (batch_size, max_num_frames, n_mfcc_features)
				Returns:
						loss: A 0-d tensor (scalar)
				"""
				loss = tf.losses.mean_squared_error(pred, self.labels_placeholder) 
				loss = tf.reduce_mean(loss)

				return loss 


		def add_training_op(self, loss):
				"""Sets up the training Ops.

				Creates an optimizer and applies the gradients to all trainable variables.
				The Op returned by this function is what must be passed to the
				`sess.run()` call to cause the model to train. See

				https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

				for more information.

				Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
										Calling optimizer.minimize() will return a train_op object.

				Args:
						loss: Loss tensor, from cross_entropy_loss.
				Returns:
						train_op: The Op for training.
				"""
				train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
				return train_op

		
		def output_wave_files(self, predicted_batch, true_target_batch, true_source_batch):
				"""Outputs and saves a single batch of wavefiles from their MFCC features. 

				Args:
					predicted_mfccs_batch: A np.ndarray (Tensorflow evaluated tensor) of shape 
						(batch_size, max_num_frames, num_mfcc_coeffs)
					true_target_mfccs_batch: A np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
				"""
				# Only outputting 1 wavefile in the batch, because otherwise it takes too long
				for i in range(min(1, predicted_batch.shape[0])):
					print "Converting wavefile ", i
					predicted = predicted_batch[i,:]
					target = true_target_batch[i,:]
					source = true_source_batch[i,:]
					wav.write('wav2wav_predicted_'+str(i)+'.wav', self.config.sample_rate, predicted)
					wav.write('wav2wav_target_'+str(i)+'.wav', self.config.sample_rate, target)
					wav.write('wav2wav_source_'+str(i)+'.wav', self.config.sample_rate, source)
					print 'wrote to predicted', i
					print 'wrote to target', i
					print 'wrote to source', i


		def train_on_batch(self, sess, inputs_batch, labels_batch, should_output_wavefiles):
				"""Perform one step of gradient descent on the provided batch of data.

				Args:
						sess: tf.Session()
						input_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
						input_masks_batch: np.ndarray of shape (batch_size, max_num_frames)
						labels_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
						label_masks_batch: np.ndarray of shape (batch_size, max_num_frames)
						should_output_wavefiles: bool that specifies whether or not we should output wavefiles from the predicted MFCC features
				Returns:
						loss: loss over the batch (a scalar)
						summary: to be used for Tensorboard
				"""
				feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
				_, loss, summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed)

				# We only evaluate the first batch in the epoch
				if should_output_wavefiles:
					predicted_batch = self.predicted.eval(session=sess, feed_dict=feed)
					true_target_batch = self.labels_placeholder.eval(session=sess, feed_dict=feed)
					true_source_batch = self.input_placeholder.eval(session=sess, feed_dict=feed)
					self.output_wave_files(predicted_batch, true_target_batch, true_source_batch)

				return loss, summary 


		def run_epoch(self, sess, inputs, labels, train_writer, step_i, should_output_wavefiles):
				"""Runs an epoch of training.

				Args:
						sess: tf.Session() object
						inputs: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs) 
						input_masks: A list of length num_examples with boolean np.darray entries of shape (max_num_frames,)
						labels: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs)	
						label_masks: A list of length num_examples with boolean np.darray entries of shape (max_num_frames,)
						train_writer: a tf.summary.FileWriter object
						step_i: The global number of steps taken so far (i.e., batches we've done a full forward
										and backward pass on) 
				Returns:
						average_loss: scalar. Average minibatch loss of model on epoch.
						step_i: The global number of steps taken so far (i.e., batches we've done a full forward
										and backward pass on)
				"""
				n_minibatches, total_loss = 0, 0
				for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
						# We only evaluate and output wavefiles on the first batch of the epoch
						should_output_wavefiles_batch = False
						if n_minibatches == 0: 
							should_output_wavefiles_batch = True 
						batch_loss, summary = self.train_on_batch(sess, input_batch, labels_batch, should_output_wavefiles and should_output_wavefiles_batch)
						total_loss += batch_loss

						n_minibatches += 1
						train_writer.add_summary(summary, step_i)
						step_i += 1

				return total_loss / n_minibatches, step_i 


		def optimize(self, sess, inputs, labels):
				"""Fit model on provided data.

				Args:
						sess: tf.Session()
						inputs: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs) 
						input_masks: A list of length num_examples with boolean np.darray entries of shape (max_num_frames,)
						labels: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs)	
						label_masks: A list of length num_examples with boolean np.darray entries of shape (max_num_frames,) 
				Returns:
						losses: list of losses per epoch
				"""
				train_writer = tf.summary.FileWriter(self.config.logs_path + '/train', sess.graph)
				step_i = 0

				losses = []

				for epoch in range(self.config.n_epochs):
						start_time = time()

						should_output_wavefiles_epoch = False
						if epoch % 50 == 0:
							should_output_wavefiles_epoch = True

						average_loss, step_i = self.run_epoch(sess, inputs, labels, train_writer, step_i, should_output_wavefiles_epoch)
						duration = time() - start_time
						print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
						losses.append(average_loss)

				return losses

	

		def add_summary_op(self):
				return tf.summary.merge_all()
	

		def __init__(self, config):
				"""Initializes the model.

				Args:
						config: A model configuration object of type Config
				"""
				self.config = config
				self.build()


		def build(self):
				self.add_placeholders()
				self.pred = self.add_prediction_op() 
				self.loss = self.add_loss_op(self.pred)
				tf.summary.scalar("loss", self.loss)
				self.train_op = self.add_training_op(self.loss)
				self.merged_summary_op = self.add_summary_op()


		def pad_sequence(self, features):
				"""
				Args:
					mfcc_features: a np.ndarray array of shape (num_frames, num_mfcc_coeffs)
					max_num_frames: the maximum length to which the array should be truncated or zero-padded 

				Returns:
					padded_mfcc_features: a np.ndarray of shape (max_num_frames, num_mfcc_coeffs)
					mask: a np.ndarray of shape (max_num_frames,)
				"""
				num_frames = features.shape[0]

				res = np.zeros(self.config.wav_len)
				collapsed = list()
				for i in range(0, self.config.wav_len-1):
					if i < len(features):
						res[i] = features[i]
					else:
						break

				return res


		def preprocess_data(self, config):
				"""Processes the training data and returns MFCC vectors for all of them.
				Args:
					config: the Config object with various parameters specified
				Returns:
					train_data:	A list of tuples, one for each training example: (accent 1 padded MFCC frames, accent 1 mask)
					train_labels: A list of tuples, one for each training example: (accent 2 padded MFCC frames, accent 2 mask)
				"""
				inputs = [] 
				labels = []	

				TARGET_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'	
				SOURCE_DIR = '../data/cmu_arctic/scottish-english-male-awb/reconstructed_wav/'
				index = 0
				for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
					if index >= 10:
						break
					index += 1

					if source_fname == '.DS_Store' or target_fname == '.DS_Store':
						continue

					(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
					(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)


					# Aligns the MFCC features matrices using FastDTW.
					source_features, target_features = get_dtw_series(source_wav_data, target_wav_data)

					# Pads the MFCC feature matrices (rows) to length config.max_num_frames
					source_padded_frames = self.pad_sequence(source_features)
					target_padded_frames = self.pad_sequence(target_features)


					inputs.append(source_padded_frames)
					labels.append(target_padded_frames) 

				return inputs, labels


def main():
	"""Main entry met`hod for this file."""
	config = Config()

	# Tell TensorFlow that the model will be built into the default Graph.
	# (not required but good practice)
	logs_path = config.logs_path

	with tf.Graph().as_default():
			# Build the model and add the variable initializer Op
			model = Wav2Wav(config)
			init = tf.global_variables_initializer()

			print "Preprocessing data ..."
			inputs, labels = model.preprocess_data(config)
			print "Finished preprocessing data"

			# Create a session for running Ops in the Graph
			with tf.Session() as sess:
					# Run the Op to initialize the variables 
					sess.run(init)
					# Fit the model
					losses = model.optimize(sess, inputs, labels)


if __name__ == "__main__":
		main()

