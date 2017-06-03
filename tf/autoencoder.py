import tensorflow as tf
import numpy as np
from utils.general_utils import batch_multiply_by_matrix 
from utils.fast_dtw import get_dtw_series 
from utils.pad_sequence import pad_sequence
from time import time, gmtime, strftime
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import math
import random
import copy
import matlab.engine


print "Starting matlab ... type in your password if prompted"
eng = matlab.engine.start_matlab()
eng.addpath('../invMFCCs_new')
print "Done starting matlab"

def get_minibatches(data, size):
    copied = copy.deepcopy(data)
    minibatches = list()
    random.shuffle(copied)
    i = 0
    while i < len(copied):
        minibatch = copied[i:i+size]
        minibatches.append(minibatch)
        i += size
    return minibatches


def encode(current_input, max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2):
    xavier = tf.contrib.layers.xavier_initializer(uniform=True)
    W1 = tf.get_variable("W1", shape=(num_mfcc_coeffs, state_size_1), initializer=xavier) 
    W2 = tf.get_variable("W2", shape=(state_size_1, state_size_2), initializer=xavier) 
   
    h1 = tf.tanh(batch_multiply_by_matrix(batch=current_input, matrix=W1))
    h2 = tf.tanh(batch_multiply_by_matrix(batch=h1, matrix=W2))

    encoder = [W1, W2]
    return encoder, h2


def decode(encoder, current_input, max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2):
    encoder.reverse()
    W1 = tf.transpose(encoder[0])
    W2 = tf.transpose(encoder[1])
    h1 = tf.tanh(batch_multiply_by_matrix(batch=current_input, matrix=W1))
    h2 = batch_multiply_by_matrix(batch=h1, matrix=W2)
    
    return h2


def corrupt(x):
    #return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), tf.float32))
    return x # TODO

def output_wave_files(predicted_mfccs_batch, true_target_mfccs_batch):
    """Outputs and saves a single batch of wavefiles from their MFCC features. 

    Args:
        predicted_mfccs_batch: A np.ndarray (Tensorflow evaluated tensor) of shape 
            (batch_size, max_num_frames, num_mfcc_coeffs)
        true_target_mfccs_batch: A np.ndarray of shape (batch_size, max_num_frames, num_mfcc_coeffs)
    """
    # only outputting 1 wavefile in the batch, because otherwise it takes too long
    for i in range(min(1, predicted_mfccs_batch.shape[0])):
        print "Converting wavefile ", i
        predicted_mfccs = predicted_mfccs_batch[i,:,:]
        target_mfccs = true_target_mfccs_batch[i,:,:]
        output_wave_file(predicted_mfccs, filename='autoencoder_pred_' + str(i))   
        output_wave_file(target_mfccs, filename='autoencoder_input_' + str(i))


def output_wave_file(predicted_mfccs, filename):
    """Outputs and saves a single wavefile from its MFCC features. 

    Args:
        predicted_mfccs: A np.ndarray (Tensorflow evaluated tensor) of shape 
            (max_num_frames, num_mfcc_coeffs)
    """
    global eng
    predicted_mfccs_transposed = np.transpose(predicted_mfccs)

    # MFCC features need to be a numpy array of shape (num_coefficients x num_frames) in order to be passed to the invmelfcc function
    inverted_wav_data = eng.invmelfcc(matlab.double(predicted_mfccs_transposed.tolist()), 16000.0, 25, 100.0, 0.005, 0.005)

    inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

    # scales the waveform to be between -1 and 1
    maxVec = np.max(inverted_wav_data)
    minVec = np.min(inverted_wav_data)
    inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2

    wav.write(filename + '.wav', 16000.0, inverted_wav_data)


def autoencoder(max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2):
    x = tf.placeholder(tf.float32, (None, max_num_frames, num_mfcc_coeffs))
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    encoder, output = encode(current_input, max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2)
    
    decoded = decode(encoder, output, max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2)
    cost = tf.sqrt(tf.reduce_mean(tf.square(decoded - x)))

    output_wave_files(decoded, x)
    return {'input': x, 'output': output, 'decoded': decoded, 'corrupt_prob': corrupt_prob, 'cost': cost}
    

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

    padded_mfcc_features = np.zeros((max_num_frames, num_mfcc_coeffs))

    # Truncate (or fill exactly
    if num_frames >= max_num_frames:
        padded_mfcc_features = mfcc_features[0:max_num_frames,:]

    # Append 0 MFCC vectors
    elif num_frames < max_num_frames:
        delta = max_num_frames - num_frames
        zeros = np.zeros((delta, num_mfcc_coeffs))
        padded_mfcc_features = np.concatenate((mfcc_features, zeros), axis=0)

    return padded_mfcc_features


def preprocess_data(num_mfcc_coeffs, num_filters, window_len, window_step, max_num_frames):
    """Processes the training data and returns MFCC vectors for all of them.
    Args:
        config: the Config object with various parameters specified
    Returns:
        train_data: A list of tuples, one for each training example: (accent 1 padded MFCC frames, accent 1 mask)
        train_labels: A list of tuples, one for each training example: (accent 2 padded MFCC frames, accent 2 mask)
    """
    inputs = [] 
    labels = [] 
    input_masks = []
    label_masks = []
    
    SOURCE_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'    
    TARGET_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
    index = 0
    for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
        if index >= 20:
            break
        index += 1

        if source_fname == '.DS_Store' or target_fname == '.DS_Store':
            continue

        (source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
        (target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

        source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=num_mfcc_coeffs, nfilt=num_filters, winlen=window_len, winstep=window_step))
        target_mfcc_features = np.array(mfcc(target_wav_data, samplerate=target_sample_rate, numcep=num_mfcc_coeffs, nfilt=num_filters, winlen=window_len, winstep=window_step))

        # align with FastDTW
        source_mfcc_features, target_mfcc_features = get_dtw_series(source_mfcc_features, target_mfcc_features)

        # pad MFCC feature matrices (rows) to max_num_frames
        source_padded_frames = pad_sequence(source_mfcc_features, max_num_frames)
        target_padded_frames = pad_sequence(target_mfcc_features, max_num_frames)

        inputs.append(source_padded_frames) 
        labels.append(target_padded_frames) 

    return inputs, labels


def autoencode():
    batch_size = 5
    n_epochs = 10000
    lr = 1e-3
    momentum = 0.3
    max_num_frames = 1220  # max length of warped time series
    num_mfcc_coeffs = 25
    sample_rate = 16000.0
    num_filters = 100
    window_len = 0.005   # 5 ms
    window_step = 0.005  # 5 ms 
    num_features = max_num_frames * num_mfcc_coeffs 
    state_size_1 = 50 
    state_size_2 = 50 
    epochs = 10000
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    accent_a, accent_b = preprocess_data(num_mfcc_coeffs, num_filters, window_len, window_step, max_num_frames)
    ae = autoencoder(max_num_frames, num_mfcc_coeffs, state_size_1, state_size_2)

    optimizer = tf.train.AdamOptimizer(lr).minimize(ae['cost'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # learn weights for the first accent
    for epoch_i in range(epochs):
        n_minibatches, total_loss = 0, 0
        minibatches = get_minibatches(accent_a, batch_size)
        for batch in minibatches:
            sess.run(optimizer, feed_dict={ae['input']: batch, ae['corrupt_prob']: [1.0]})
            total_loss += ae['cost'].eval(session=sess, feed_dict={ae['input']: batch, ae['corrupt_prob']: [1.0]})
            n_minibatches += 1
        print epoch_i, total_loss/n_minibatches

    # learn weights for the second accent
    for epoch_i in range(epochs):
        n_minibatches, total_loss = 0, 0
        minibatches = get_minibatches(accent_a, batch_size)
        for batch in minibatches:
            sess.run(optimizer, feed_dict={ae['input']: batch, ae['corrupt_prob']: [1.0]})
            total_loss += ae['cost'].eval(session=sess, feed_dict={ae['input']: batch, ae['corrupt_prob']: [1.0]})
            n_minibatches += 1
        print epoch_i, total_loss/n_minibatches

autoencode()











