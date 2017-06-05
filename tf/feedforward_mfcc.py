from time import time, gmtime, strftime
import numpy as np
import os
import random
import matlab.engine
import math
import datetime

import scipy.io.wavfile as wav
from python_speech_features import mfcc
from mcd.metrics import logSpecDbDist
import sounddevice as sd

import tensorflow as tf
from tensorflow.python.ops.nn import dynamic_rnn

from utils.general_utils import get_minibatches, batch_multiply_by_matrix 
from utils.fast_dtw import get_dtw_series 
from utils.pad_sequence import pad_sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_metrics(metrics_dict):
	train_mcd, = plt.plot(metrics_dict['train'], label='train')
	val_mcd, = plt.plot(metrics_dict['val'], label='val')
	plt.title('MCD Score')
	plt.legend(handles=[train_mcd, val_mcd])
	ts = time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
	plt.savefig('metric_plots/mcd_' + st)
	plt.clf()



class Config(object):
		"""Holds model hyperparams and data information.

		The config class is used to store various hyperparameters and dataset
		information parameters. Model objects are passed a Config() object at
		instantiation.
		"""
		batch_size = 5
		n_epochs = 100000
		lr = 1e-2
		momentum = 0.3

		max_num_frames = 1220  # This is the maximum length of any warped time series in the dataset 

		num_mfcc_coeffs = 25
		sample_rate = 16000.0
		num_filters = 100
		window_len = 0.005	 # 5 ms
		window_step = 0.005  # 5 ms	

		num_features = max_num_frames * num_mfcc_coeffs 
		state_size_1 = 50 
		state_size_2 = 50 
		#state_size_3 = 50
		#state_size_4 = 25
		dropout_keep_prob = 0.8
		logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

		val_size = 200	# Size of validation set


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
				self.input_placeholder = tf.placeholder(tf.float32, (None, self.config.max_num_frames, self.config.num_mfcc_coeffs))
				self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.max_num_frames, self.config.num_mfcc_coeffs))
				self.dropout_placeholder = tf.placeholder(tf.float32, ())


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
					self.labels_placeholder: labels_batch,
					self.dropout_placeholder: self.config.dropout_keep_prob
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
				W1 = tf.get_variable("W1", shape=(self.config.num_mfcc_coeffs, self.config.state_size_1), initializer=xavier) 
				W2 = tf.get_variable("W2", shape=(self.config.state_size_1, self.config.state_size_2), initializer=xavier) 
				W3 = tf.get_variable("W3", shape=(self.config.state_size_2, self.config.num_mfcc_coeffs), initializer=xavier) 

				# [batch, max_num_frames, num_mfcc_coeffs] x [num_mfcc_coeffs, state_size1] = [batch, max_num_frames, state_size1]
				print "inputs shape: ", self.input_placeholder
				h1 = tf.tanh(batch_multiply_by_matrix(batch=self.input_placeholder, matrix=W1))
				print "h1 shape: ", h1

				# [batch, max_num_frames, state_size1] x [state_size1, state_size2] = [batch, max_num_frames, state_size2]
				h2 = tf.tanh(batch_multiply_by_matrix(batch=h1, matrix=W2))
				print "h2 shape: ", h2
				
				# [batch, max_num_frames, state_size2] x [state_size2, num_mfcc_coeffs] = [batch, max_num_frames, num_mfcc_coeffs]
				mfcc_preds = batch_multiply_by_matrix(batch=h2, matrix=W3) 
				print "mfcc preds shape: ", mfcc_preds

				self.mfcc = mfcc_preds
				return mfcc_preds 
 

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


		def mcd_score(self, features, true_features):
				"""Evaluates the MCD score for two aligned MFCC feature matrices.
	
				Args:
					features: np.ndarray of shape (max_num_frames, num_mfcc_coeffs)
					true_features: np.ndarray of shape (max_num_frames, num_mfcc_coeffs)

				Returns:
					score: A float representing the MCD score.
				"""
				logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)
				diff = features - true_features
				mcd_cols = np.sqrt( (diff*diff).sum(axis=0) )
				score = np.sum(mcd_cols)
				return score


		def output_wave_files(self, predicted_mfccs_batch, true_target_mfccs_batch):
				"""Outputs and saves a single batch of wavefiles from their MFCC features. 

				Args:
					predicted_mfccs_batch: A np.ndarray (Tensorflow evaluated tensor) of shape 
						(batch_size, max_num_frames, num_mfcc_coeffs)
					true_target_mfccs_batch: A np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
				"""
				# Only outputting 1 wavefile in the batch, because otherwise it takes too long
				for i in range(min(1, predicted_mfccs_batch.shape[0])):
					print "Converting wavefile ", i
					predicted_mfccs = predicted_mfccs_batch[i,:,:]
					target_mfccs = true_target_mfccs_batch[i,:,:]
					self.output_wave_file(predicted_mfccs, filename='predicted_wav' + str(i))	
					self.output_wave_file(target_mfccs, filename='true_wav' + str(i))


		def output_wave_file(self, predicted_mfccs, filename='predicted_wav'):
				"""Outputs and saves a single wavefile from its MFCC features. 

				Args:
					predicted_mfccs: A np.ndarray (Tensorflow evaluated tensor) of shape 
						(max_num_frames, num_mfcc_coeffs)
				"""
				predicted_mfccs_transposed = np.transpose(predicted_mfccs)

				# MFCC features need to be a numpy array of shape (num_coefficients x num_frames) in order to be passed to the invmelfcc function
				inverted_wav_data = self.eng.invmelfcc(matlab.double(predicted_mfccs_transposed.tolist()),
																							 self.config.sample_rate,
																							 self.config.num_mfcc_coeffs,
																							 float(self.config.num_filters), self.config.window_len, self.config.window_step)

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
					predicted_mfccs_batch = self.mfcc.eval(session=sess, feed_dict=feed)
					true_target_mfccs_batch = self.labels_placeholder.eval(session=sess, feed_dict=feed)
					print "Predicted mfcc: ", predicted_mfccs_batch[0]
					print "Target mfcc: ", true_target_mfccs_batch[0]
					self.output_wave_files(predicted_mfccs_batch, true_target_mfccs_batch)

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
						batch_loss, summary = self.train_on_batch(sess, input_batch, 
																														labels_batch, 
																														should_output_wavefiles and should_output_wavefiles_batch)
						total_loss += batch_loss

						n_minibatches += 1
						train_writer.add_summary(summary, step_i)
						step_i += 1

				return total_loss / n_minibatches, step_i 


		def optimize(self, sess, train_inputs, train_labels, val_inputs, val_labels):
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

				train_val_mcds = {'train': [], 'val': []} 

				for epoch in range(self.config.n_epochs):
						start_time = time()

						should_output_wavefiles_epoch = False
						if (epoch > 0) and (epoch % 50 == 0):
							should_output_wavefiles_epoch = True
							train_mcd = self.validate(sess, train_inputs, train_labels)
							val_mcd = self.validate(sess, val_inputs, val_labels) 
							print "train mcd: ", train_mcd
							print "val mcd: ", val_mcd
							train_val_mcds['train'].append(train_mcd)
							train_val_mcds['val'].append(val_mcd)
							plot_metrics(train_val_mcds)

						average_loss, step_i = self.run_epoch(sess, train_inputs, train_labels, train_writer, step_i, 
																									should_output_wavefiles_epoch)
					
						# Save model
						self.saver.save(sess, "saved_models/mfcc_model_epoch_" + str(epoch))

						duration = time() - start_time
						print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)


		def evaluate_batch(self, sess, input_batch, labels_batch):
				feed = {}
				feed[self.input_placeholder] = input_batch
				feed[self.labels_placeholder] = labels_batch
				# No dropout on test 
				feed[self.dropout_placeholder] = 1.0

				_, loss, summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed)

				features_batch = self.mfcc.eval(feed_dict=feed, session=sess)
				true_features_batch = labels_batch

				mcds_for_batch = []
				for i in range(len(true_features_batch)):
					features_single_ex = features_batch[i,:,:]
					true_features_single_ex = true_features_batch[i]
					mcd_one_example = self.mcd_score(features_single_ex, true_features_single_ex)
					mcds_for_batch.append(mcd_one_example)

				return mcds_for_batch	


		def validate(self, sess, inputs, labels, model_dir=None, model_name=None):
				"""Evaluates the provided model on the provided validation data.
				
				Returns:
					mcd: the MCD score for every input
				"""
				# Only need to restore model if we're not validating during training
				if (model_dir is not None) and (model_name is not None): 
					self.saver = tf.train.import_meta_graph(model_path + model_name + ".meta")
					self.saver.restore(sess, model_path + model_name)

				mcds = []
				step = 50

				for start_idx in range(0, len(inputs), step):
					end_idx = min(start_idx + step, len(inputs))
					mcds_one_batch = self.evaluate_batch(sess, inputs[start_idx:end_idx], labels[start_idx:end_idx])
					mcds += mcds_one_batch
				mcd_total = sum(mcds) / float(len(mcds))
				return mcd_total

	
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
				self.eng = self.setup_matlab_engine()
				self.mfcc = None	# Add a handle to this so we can set it later
				self.saver = None # Model saver
				self.add_placeholders()
				self.pred = self.add_prediction_op() 
				self.loss = self.add_loss_op(self.pred)
				self.train_op = self.add_training_op(self.loss)

				tf.summary.scalar("loss", self.loss)
				self.merged_summary_op = self.add_summary_op()


		def preprocess_data(self, config):
				"""Processes the training data and returns MFCC vectors for all of them.
				Args:
					config: the Config object with various parameters specified
				Returns:
					train_data:	A list of features, one for each training example: accent 1 padded MFCC frames
					train_labels: A list of features, one for each training example: accent 2 padded MFCC frames
				"""
				inputs = [] 
				labels = []	
				
				SOURCE_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'	
				TARGET_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
				#TARGET_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'	
				#TARGET_DIR = '../data/cmu_arctic/indian-english-male-ksp/wav/'
				index = 0
				for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
					#if index >= 20:
					#	break
					#index += 1

					if source_fname == '.DS_Store' or target_fname == '.DS_Store':
						continue

					(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
					(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

					# appendEnergy is False because we want to keep the 0th coefficient
					source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=self.config.num_mfcc_coeffs,
																							 nfilt=self.config.num_filters, winlen=self.config.window_len, winstep=self.config.window_step, appendEnergy=False))
					target_mfcc_features = np.array(mfcc(target_wav_data, samplerate=target_sample_rate, numcep=self.config.num_mfcc_coeffs,
																							 nfilt=self.config.num_filters, winlen=self.config.window_len, winstep=self.config.window_step, appendEnergy=False))

					# Aligns the MFCC features matrices using FastDTW.
					source_mfcc_features, target_mfcc_features = get_dtw_series(source_mfcc_features, target_mfcc_features)

					# Pads the MFCC feature matrices (rows) to length config.max_num_frames
					source_padded_frames, _ = pad_sequence(source_mfcc_features, config.max_num_frames)
					target_padded_frames, _ = pad_sequence(target_mfcc_features, config.max_num_frames)

					#if index < 20:
					#	self.output_wave_file(source_padded_frames, filename='src' + str(index))
					#	self.output_wave_file(target_padded_frames, filename='tgt' + str(index))
						#wav.write('source' + str(index) + '.wav', self.config.sample_rate, source_wav_data)
						#wav.write('target' + str(index) + '.wav', self.config.sample_rate, target_wav_data)
						#self.eng.soundsc(matlab.double(source_wav_data.tolist()), self.config.sample_rate, nargout=0)
						#self.eng.soundsc(matlab.double(target_wav_data.tolist()), self.config.sample_rate, nargout=0)
					#index += 1

					inputs.append(source_padded_frames) 
					labels.append(target_padded_frames) 

				return inputs, labels 



def run(train=True):
	"""Main entry point for this file."""
	config = Config()

	# Tell TensorFlow that the model will be built into the default Graph.
	# (not required but good practice)
	logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

	with tf.Graph().as_default():
			model = ANNModel(config)
			init = tf.global_variables_initializer()
			model.saver = tf.train.Saver()

			print "Preprocessing data ..."
			inputs, labels = model.preprocess_data(config)
			print "Finished preprocessing data"

			# Create a session for running Ops in the Graph
			with tf.Session() as sess:
				# Run the Op to initialize the variables 
				sess.run(init)

				train_inputs = inputs[0:len(inputs)-model.config.val_size] # Gets all but the last <val_size> examples 
				train_labels = labels[0:len(labels)-model.config.val_size]
				val_inputs = inputs[len(inputs)-model.config.val_size:]
				val_labels = labels[len(labels)-model.config.val_size:]

				if train:
					model.optimize(sess, train_inputs, train_labels, val_inputs, val_labels)

				else: # Validate
					model_dir = "saved_models"
					model_name = ""
					model.validate(sess, train_inputs, train_labels, model_dir, model_name)


if __name__ == "__main__":
		run(train=True)

