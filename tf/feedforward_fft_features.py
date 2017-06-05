from time import time, gmtime, strftime, sleep
import numpy as np
from numpy.fft import fft, ifft
import os
import random
import matlab.engine

import scipy.io.wavfile as wav
from python_speech_features import mfcc
import sounddevice as sd

import tensorflow as tf
from tensorflow.python.ops.nn import dynamic_rnn

from utils.general_utils import get_minibatches, batch_multiply_by_matrix 
from utils.fast_dtw import get_dtw_series 
from utils.pad_sequence import pad_sequence


class Config(object):
		"""Holds model hyperparams and data information.

		The config class is used to store various hyperparameters and dataset
		information parameters. Model objects are passed a Config() object at
		instantiation.
		"""
		batch_size = 5
		n_epochs = 100000
		lr = 1e-1
		momentum = 0.3

		sample_rate = 16000.0
		num_filters = 100
		window_len = 0.005	 # 5 ms
		window_step = 0.005  # 5 ms	

		max_num_samples = 85120
		num_samples_per_frame = int(sample_rate * window_len)  # = 80 samples/frame 
		max_num_frames = max_num_samples / num_samples_per_frame	 # = 1064 because the max recording has 85120 samples, and 85120 / 80 = 1064 

		num_features = max_num_frames * num_samples_per_frame 
		state_size_1 = 50 
		state_size_2 = 50 
		dropout_keep_prob = 1.0 # 0.8
		logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


class ANNModel(object):
		"""Implements a feedforward neural network with a MSE loss."""

		def add_placeholders(self):
				"""Generates placeholder variables to represent the input tensors.

				These placeholders are used as inputs by the rest of the model building
				and will be fed data during training.

				Adds following nodes to the computational graph

				input_placeholder: Input placeholder tensor of shape
													 (batch_size, max_num_frames, num_mfcc_coeffs), type tf.complex64
				labels_placeholder: Labels placeholder tensor of shape
														(batch_size, max_num_frames, num_mfcc_coeffs), type tf.complex64
				"""
				self.input_placeholder = tf.placeholder(tf.complex64, (None, self.config.max_num_frames, self.config.num_samples_per_frame))
				self.labels_placeholder = tf.placeholder(tf.complex64, (None, self.config.max_num_frames, self.config.num_samples_per_frame))
				self.dropout_placeholder = tf.placeholder(tf.float32, ())


		def create_feed_dict(self, inputs_batch, labels_batch=None):
				"""Creates the feed_dict for training the given step.
				Args:
						inputs_batch: A batch of input data.
						labels_batch: (Optional) a batch of label data.
				Returns:
						feed_dict: The feed dictionary mapping from placeholders to values.
				"""
				feed_dict = {
					self.input_placeholder: inputs_batch,
					self.labels_placeholder: labels_batch,
					self.dropout_placeholder: self.config.dropout_keep_prob
				}
				return feed_dict


		def complex_to_float_tensor(self, input_tensor):
			# Concatenates the complex component after the real component along the last axis.
			return tf.concat( [tf.real(input_tensor), tf.imag(input_tensor)], axis=-1)


		def add_prediction_op(self): 
				"""Adds the core transformation for this model which transforms a batch of input
				data into a batch of predictions. 
			
				The network is a 3-layer ANN with weights and no biases.	

				Returns:
						pred: A tensor of shape (batch_size, max_num_frames, 2*num_samples_per_frame)
				"""

				xavier = tf.contrib.layers.xavier_initializer()

				# It's 2 * num_samples_per_frame because we're dealing with complex numbers
				W1 = tf.get_variable("W1", shape=(2*self.config.num_samples_per_frame, self.config.state_size_1), dtype=tf.float32, initializer=xavier) 
				W2 = tf.get_variable("W2", shape=(self.config.state_size_1, self.config.state_size_2), dtype=tf.float32, initializer=xavier) 
				W3 = tf.get_variable("W3", shape=(self.config.state_size_2, 2*self.config.num_samples_per_frame), dtype=tf.float32, initializer=xavier) 

				# [batch, max_num_frames, 2*num_samples_per_frame] x [2*num_samples_per_frame, state_size1] = [batch, max_num_frames, state_size1]
				print "inputs shape: ", self.input_placeholder
				twice_as_long_input = self.complex_to_float_tensor(self.input_placeholder)
				print "twice_as_long_input shape: ", twice_as_long_input
				h1 = tf.tanh(batch_multiply_by_matrix(batch=twice_as_long_input, matrix=W1))
				print "h1 shape: ", h1

				# [batch, max_num_frames, state_size1] x [state_size1, state_size2] = [batch, max_num_frames, state_size2]
				h2 = tf.tanh(batch_multiply_by_matrix(batch=h1, matrix=W2))
				print "h2 shape: ", h2
				
				# [batch, max_num_frames, state_size2] x [state_size2, 2*num_samples_per_frame] = [batch, max_num_frames, 2*num_samples_per_frame]
				fft_preds_real_2x = batch_multiply_by_matrix(batch=h2, matrix=W3) 
				print "fft preds real 2x shape: ", fft_preds_real_2x

				# Convert back into complex numbers
				fft_preds_reals = tf.slice(fft_preds_real_2x, [0, 0, 0], 
																											[-1, self.config.max_num_frames, self.config.num_samples_per_frame]) 
				fft_preds_complexes = tf.slice(fft_preds_real_2x, [0, 0, self.config.num_samples_per_frame],
																													[-1, self.config.max_num_frames, self.config.num_samples_per_frame])
				self.fft = tf.complex(fft_preds_reals, fft_preds_complexes)
				print "fft preds complex shape: ", self.fft 

				# Return the twice as long real-valued tensor
				return fft_preds_real_2x 
 

		def add_loss_op(self, pred_real_2x):
				"""Adds mean squared error ops to the computational graph.

				Args:
						pred: A tensor of shape (batch_size, max_num_frames, 2*num_samples_per_frame)
				Returns:
						loss: A 0-d tensor (scalar)
				"""
				loss = tf.losses.mean_squared_error(pred_real_2x, self.complex_to_float_tensor(self.labels_placeholder)) 
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

		
		def output_wave_files(self, predicted_ffts_batch, true_target_ffts_batch):
				"""Outputs and saves a single batch of wavefiles from their MFCC features. 

				Args:
					predicted_mfccs_batch: A np.ndarray (Tensorflow evaluated tensor) of shape 
						(batch_size, max_num_frames, num_mfcc_coeffs)
					true_target_mfccs_batch: A np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
				"""
				# Only outputting 1 wavefile in the batch, because otherwise it takes too long
				for i in range(min(1, predicted_ffts_batch.shape[0])):
					print "Converting wavefile ", i
					predicted_ffts = predicted_ffts_batch[i,:,:]
					target_ffts = true_target_ffts_batch[i,:,:]
					self.output_wave_file(predicted_ffts, filename='predicted_wav' + str(i))	
					self.output_wave_file(target_ffts, filename='true_wav' + str(i))


		def output_wave_file(self, predicted_ffts, filename='predicted_wav'):
				"""Outputs and saves a single wavefile from its MFCC features. 

				Args:
					predicted_mfccs: A np.ndarray (Tensorflow evaluated tensor) of shape 
						(max_num_frames, num_mfcc_coeffs)
				"""
				# Concatenate all the FFT values into a 1D array
				predicted_ffts = np.reshape(predicted_ffts, (-1,))	# Concatenate all the fft values together

				# Get rid of the trailing zeros 
				predicted_ffts = np.trim_zeros(predicted_ffts, 'b')

				# Do the inverse FFT to the predicted data, and only consider its real values for playback
				inverted_wav_data = ifft(predicted_ffts) 
				inverted_wav_data = inverted_wav_data.real

				sd.play(inverted_wav_data, self.config.sample_rate)
				inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

				# Scales the waveform to be between -1 and 1
				maxVec = np.max(inverted_wav_data)
				minVec = np.min(inverted_wav_data)
				inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2

				wav.write(filename + '.wav', self.config.sample_rate, inverted_wav_data)


		def train_on_batch(self, sess, inputs_batch, labels_batch, should_output_wavefiles):
				"""Perform one step of gradient descent on the provided batch of data.

				Args:
						sess: tf.Session()
						input_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
						labels_batch: np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
						should_output_wavefiles: bool that specifies whether or not we should output wavefiles from the predicted MFCC features
				Returns:
						loss: loss over the batch (a scalar)
						summary: to be used for Tensorboard
				"""
				feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
				_, loss, summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed)

				# We only evaluate the first batch in the epoch
				if should_output_wavefiles:
					predicted_ffts_batch = self.fft.eval(session=sess, feed_dict=feed)
					true_target_ffts_batch = self.labels_placeholder.eval(session=sess, feed_dict=feed)
					print "Predicted ffts: ", predicted_ffts_batch[0]
					print "Target ffts: ", true_target_ffts_batch[0]
					self.output_wave_files(predicted_ffts_batch, true_target_ffts_batch)

				return loss, summary 


		def run_epoch(self, sess, inputs, labels, train_writer, step_i, should_output_wavefiles):
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
				for input_batch, labels_batch in \
										get_minibatches([inputs, labels], self.config.batch_size):

						# We only evaluate and output wavefiles on the first batch of the epoch
						should_output_wavefiles_batch = False
						if n_minibatches == 0: 
							should_output_wavefiles_batch = True 
						batch_loss, summary = self.train_on_batch(sess, input_batch, labels_batch, 
																														should_output_wavefiles and should_output_wavefiles_batch)
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
						labels: A list of length num_examples with float np.ndarray entries of shape (max_num_frames, num_mfcc_coeffs)	
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

						average_loss, step_i = self.run_epoch(sess, inputs, labels, train_writer, step_i, 
																									should_output_wavefiles_epoch)
						duration = time() - start_time
						print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
						losses.append(average_loss)

				return losses


		def evaluate_batch(self, sess, input_batch, labels_batch):
				feed = {}
				feed[self.input_placeholder] = input_batch 
				feed[self.labels_placeholder] = labels_batch 
				# No dropout on test 
				feed[self.dropout_placeholder] = 1.0

				_, loss, summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed)

				features_batch = self.fft.eval(feed_dict=input_feed, session=sess)
				true_features_batch = labels_batch 
				
				mcds_for_batch = []
				for i in range(len(true_features_batch)):
					features_single_ex = features_batch[i,:,:] 
					true_features_single_ex = true_features_batch[i]
					mcd_one_example = mcd_score(features_single_ex, true_features_single_ex)
					mcds_for_batch.append(mcd_one_example)

				return mcds_for_batch 


		def validate(self, sess, inputs, labels, model_dir, model_name):
				"""Evaluates the provided model on the provided validation data.
				
				Returns:
					mcd: the MCD score for every input
				"""
				self.saver = tf.train.import_meta_graph(model_path + model_name + ".meta")
				self.saver.restore(sess, model_path + model_name)

				mcds = []
				step = 50 

				for start_idx in range(0, len(dataset), step):
					end_idx = min(start_idx + step, len(inputs))
					mcds_one_batch = self.evaluate_batch(sess, inputs[start_idx:end_idx], labels[start_idx:end_idx])
					mcds += mcds_one_batch 
				mcd_total = sum(mcds) / float(len(mcds))
				print("Total MCD: ", mcd_total)	
	

		def setup_matlab_engine(self):
				print "Starting matlab ... type in your password if prompted"
				eng = matlab.engine.start_matlab()
				eng.addpath('../invMFCCs_new')
				print "Done starting matlab"
				return eng				


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
				self.fft = None	 # Add a handle to this so we can set it later
				self.saver = None  # Model saver
				self.add_placeholders()
				self.pred = self.add_prediction_op() 
				self.loss = self.add_loss_op(self.pred)
				tf.summary.scalar("loss", self.loss)
				self.train_op = self.add_training_op(self.loss)
				self.merged_summary_op = self.add_summary_op()


		def preprocess_data(self, config):
				"""Processes the training data and returns MFCC vectors for all of them.
				Args:
					config: the Config object with various parameters specified
				Returns:
					train_data:	A list, one for each training example: accent 1 padded FFT data 
					train_labels: A list, one for each training example: accent 2 padded FFT data
				"""
				inputs = [] 
				labels = []	
				
				#SOURCE_DIR = '../data/cmu_arctic/us-english-female-slt/wav/'	
				SOURCE_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
				TARGET_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'	
				#TARGET_DIR = '../data/cmu_arctic/indian-english-male-ksp/wav/'
				index = 0
				for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
					if index >= 5:
						break
					index += 1
					if source_fname == '.DS_Store' or target_fname == '.DS_Store':
						continue
		
					(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
					(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

					src_fft = fft(source_wav_data)	# Both of these are complex numbers
					tgt_fft = fft(target_wav_data)	

					source_padded_frames = pad_sequence(src_fft, config.max_num_frames, 
								num_samples_per_frame=self.config.num_samples_per_frame)
					target_padded_frames = pad_sequence(tgt_fft, config.max_num_frames, 
								num_samples_per_frame=self.config.num_samples_per_frame)

					source_padded_frames = np.reshape(source_padded_frames, (self.config.max_num_frames, self.config.num_samples_per_frame)) 
					target_padded_frames= np.reshape(target_padded_frames, (self.config.max_num_frames, self.config.num_samples_per_frame))

					inputs.append(source_padded_frames) 
					labels.append(target_padded_frames) 

				return inputs, labels


def train():
	"""Training method for this file."""
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

			train_inputs = inputs[0:len(inputs)-200]  # Gets all but the last 200 examples 
			train_labels = labels[0:len(labels)-200]

			# Create a session for running Ops in the Graph
			with tf.Session() as sess:
					# Run the Op to initialize the variables 
					sess.run(init)
					# Fit the model
					losses = model.optimize(sess, train_inputs, train_labels)


def validate():
	config = Config()

	# Tell TensorFlow that the model will be built into the default Graph.
  # (not required but good practice)
  logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

	with tf.Graph().as_default():
      model = ANNModel(config)
      init = tf.global_variables_initializer()

      print "Preprocessing data ..."
      inputs, labels = model.preprocess_data(config)
			print "Finished preprocessing data"

			val_inputs = inputs[len(inputs)-200:]
			val_labels = labels[len(labels)-200:]

			# Create a session for running Ops in the Graph
      with tf.Session() as sess:
				# Run the Op to initialize the variables 
      	sess.run(init)

				model_dir = "saved_models"
				model_name = ""
				model.validate(sess, train_inputs, train_labels, model_dir, model_name) 


if __name__ == "__main__":
		train()

