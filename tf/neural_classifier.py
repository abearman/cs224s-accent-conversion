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
		batch_size = 5
		n_epochs = 10000
		lr = 1e-3
		momentum = 0.3

		max_num_frames = 1220  # This is the maximum length of any warped time series in the dataset 

		num_mfcc_coeffs = 25
		sample_rate = 16000.0
		num_filters = 100
		window_len = 0.005	 # 5 ms
		window_step = 0.005  # 5 ms	
		n_classes = 2

		num_features = max_num_frames * num_mfcc_coeffs 
		state_size_1 = 50 
		state_size_2 = 50 
		dropout_keep_prob = 1.0 # 0.8
		logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


class ANNModel(object):
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

				Add these placeholders to self as the instance variables
						self.input_placeholder
						self.labels_placeholder
				"""
				self.input_placeholder = tf.placeholder(tf.float32, (None, self.config.num_features))
				self.labels_placeholder = tf.placeholder(tf.int32, (None, ))



		def create_feed_dict(self, inputs_batch, labels_batch=None):
				"""Creates the feed_dict for training the given step.

				A feed_dict takes the form of:
				feed_dict = {
								<placeholder>: <tensor of values to be passed for placeholder>,
								....
				}
				
				Args:
						inputs_batch: A batch of input data.
						labels_batch: (Optional) a batch of label data.
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
			
				Returns:
						pred: A tensor of shape (batch_size, n_classes)
				"""

				xavier = tf.contrib.layers.xavier_initializer()
				W1 = tf.get_variable("W1", shape=(self.config.num_features, self.config.state_size_1), initializer=xavier) 
				b1 = tf.get_variable("b1", shape=(1, self.config.state_size_1))
				W2 = tf.get_variable("W2", shape=(self.config.state_size_1, self.config.state_size_2), initializer=xavier) 
				b2 = tf.get_variable("b2", shape=(1, self.config.state_size_2))
				W3 = tf.get_variable("W3", shape=(self.config.state_size_2, self.config.n_classes), initializer=xavier) 
				b3 = tf.get_variable("b3", shape=(1, self.config.n_classes))

				h1 = tf.tanh(tf.matmul(self.input_placeholder, W1) + b1)
				h2 = tf.tanh(tf.matmul(h1, W2) + b2)
				
				pred = tf.matmul(h2, W3) + b3
				return pred
 

		def add_loss_op(self, pred):
				"""Adds mean squared error ops to the computational graph.

				Args:
						pred: A tensor of shape (batch_size, max_num_frames, n_mfcc_features)
				Returns:
						loss: A 0-d tensor (scalar)
				"""
				loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred) 
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


		def train_on_batch(self, sess, inputs_batch, labels_batch):
				"""Perform one step of gradient descent on the provided batch of data.

				Args:
						sess: tf.Session()
						input_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
						labels_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
				Returns:
						loss: loss over the batch (a scalar)
						summary: to be used for Tensorboard
				"""
				feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
				_, loss, summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed)

				return loss, summary 


		def run_epoch(self, sess, inputs, labels, train_writer, step_i):
				"""Runs an epoch of training.

				Args:
						sess: tf.Session() object
						inputs: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs) 
						labels: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs)	
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
						batch_loss, summary = self.train_on_batch(sess, input_batch, labels_batch)
						total_loss += batch_loss

						n_minibatches += 1
						train_writer.add_summary(summary, step_i)
						step_i += 1

				return total_loss / n_minibatches, step_i 


		def optimize(self, sess, inputs, labels, test_inputs, test_labels):
				"""Fit model on provided data.

				Args:
						sess: tf.Session()
						inputs: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs) 
						labels: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs)	
				Returns:
						losses: list of losses per epoch
				"""
				train_writer = tf.summary.FileWriter(self.config.logs_path + '/train', sess.graph)
				step_i = 0

				losses = list()
				preds = list()

				print len(inputs)
				print len(test_inputs)

				for epoch in range(self.config.n_epochs):
						start_time = time()
						average_loss, step_i = self.run_epoch(sess, inputs, labels, train_writer, step_i)
						duration = time() - start_time
						print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
						losses.append(average_loss)
						if epoch % 100 == 0 and epoch != 0:	
							predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict={self.input_placeholder: test_inputs})						
							correct = 0
							encountered = len(predictions)
							for i in range(0, len(predictions)):									
								if predictions[i] == test_labels[i]:
									correct += 1
							print 'batch correct',  correct/float(encountered)




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
				self.mfcc = None	# Add a handle to this so we can set it later
				self.add_placeholders()
				self.pred = self.add_prediction_op() 
				self.loss = self.add_loss_op(self.pred)
				tf.summary.scalar("loss", self.loss)
				self.train_op = self.add_training_op(self.loss)
				self.merged_summary_op = self.add_summary_op()


		def pad_sequence(self, mfcc_features):
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

				res = np.zeros(self.config.num_features)
				collapsed = list()
				for i in range(0, num_frames):
					for j in range(0, num_mfcc_coeffs):
						res[i+j] = mfcc_features[i][j]
				
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
				
				SOURCE_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'	
				TARGET_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
				index = 0
				for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
					if index >= 500:
						break
					index += 1

					if source_fname == '.DS_Store' or target_fname == '.DS_Store':
						continue

					(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
					(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

					source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=self.config.num_mfcc_coeffs,
																							 nfilt=self.config.num_filters, winlen=self.config.window_len, winstep=self.config.window_step))
					target_mfcc_features = np.array(mfcc(target_wav_data, samplerate=target_sample_rate, numcep=self.config.num_mfcc_coeffs,
																							 nfilt=self.config.num_filters, winlen=self.config.window_len, winstep=self.config.window_step))

					# Aligns the MFCC features matrices using FastDTW.
					source_mfcc_features, target_mfcc_features = get_dtw_series(source_mfcc_features, target_mfcc_features)

					# Pads the MFCC feature matrices (rows) to length config.max_num_frames
					source_padded_frames = self.pad_sequence(source_mfcc_features)
					target_padded_frames = self.pad_sequence(target_mfcc_features)


					inputs.append(source_padded_frames)
					inputs.append(target_padded_frames)
					labels.append(0) 
					labels.append(1) 

				return inputs, labels


def main():
	"""Main entry method for this file."""
	config = Config()

	# Tell TensorFlow that the model will be built into the default Graph.
	# (not required but good practice)
	logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

	with tf.Graph().as_default():
			# Build the model and add the variable initializer Op
			model = ANNModel(config)
			init = tf.global_variables_initializer()

			print "Preprocessing data ..."
			inputs, labels = model.preprocess_data(config)
			print "Finished preprocessing data"
			all_data = len(inputs)
			split = all_data / 10
			train_inputs = inputs[split:]
			test_inputs = inputs[:split]

			train_labels = labels[split:]
			test_labels = labels[:split]

			# Create a session for running Ops in the Graph
			with tf.Session() as sess:
					# Run the Op to initialize the variables 
					sess.run(init)
					# Fit the model
					losses = model.optimize(sess, train_inputs, train_labels, test_inputs, test_labels)
					

if __name__ == "__main__":
		main()

