import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft
from utils.general_utils import batch_multiply_by_matrix, batch_multiply_by_matrix_numpy
from utils.fast_dtw import get_dtw_series 
from utils.pad_sequence import pad_sequence
from time import time, gmtime, strftime
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import math
import random
import copy
import sounddevice as sd


def get_minibatches(data_a, data_b, size):
    indexes = np.arange(0, len(data_a))
    np.random.shuffle(indexes)
    shuffled_a = list()
    shuffled_b = list()
    i = 0
    while i < len(indexes)-1:
        minibatch_a = list()
        minibatch_b = list()
        for j in range(0, size):
            if i + j >= len(indexes): # done with last minibatch
                break
            minibatch_a.append(data_a[indexes[i+j]])
            minibatch_b.append(data_b[indexes[i+j]])
        i += size
        shuffled_a.append(minibatch_a)
        shuffled_b.append(minibatch_b)

    return np.array(shuffled_a), np.array(shuffled_b)


def complex_to_float_tensor(input_tensor):
    # Concatenates the complex component after the real component along the last axis.
    return tf.concat([tf.real(input_tensor), tf.imag(input_tensor)], axis=-1)

def complex_to_float_numpy(input_array):
    res = list()
    for entry in input_array:
        res.append(np.concatenate([entry.real, entry.imag], axis=-1))
    return res

def encode(current_input, max_num_frames, num_samples_per_frame, state_size_1, state_size_2, version):
    xavier = tf.contrib.layers.xavier_initializer()

    W1 = tf.get_variable("W1"+version, shape=(2*num_samples_per_frame, state_size_1), dtype=tf.float32, initializer=xavier) 
    W2 = tf.get_variable("W2"+version, shape=(state_size_1, state_size_2), dtype=tf.float32, initializer=xavier) 
   
    h1 = tf.tanh(batch_multiply_by_matrix(batch=current_input, matrix=W1))
    h2 = tf.tanh(batch_multiply_by_matrix(batch=h1, matrix=W2))

    encoder = [W1, W2]
    return encoder, h2


def decode(encoder, current_input, max_num_frames, num_samples_per_frame, state_size_1, state_size_2):
    encoder.reverse()
    W1 = tf.transpose(encoder[0])
    W2 = tf.transpose(encoder[1])
    h1 = tf.tanh(batch_multiply_by_matrix(batch=current_input, matrix=W1))
    fft_preds_real_2x = batch_multiply_by_matrix(batch=h1, matrix=W2)

    return fft_preds_real_2x


def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.float32), tf.float32))

def collapse(multidim):
    print multidim.shape
    final = np.zeros(multidim.shape[0] * multidim.shape[1])
    for i in range(0, multidim.shape[0]):
        for j in range(0, multidim.shape[1]):
            final[i+j] = multidim[i][j]
    return final


def output_wave_file(predicted, filename):
    """Outputs and saves a single wavefile from its MFCC features. 

    Args:
        predicted_mfccs: A np.ndarray (Tensorflow evaluated tensor) of shape 
            (max_num_frames, num_mfcc_coeffs)
    """
    print 'input to output_wave_files', predicted.shape, predicted.dtype

    # Concatenate all the FFT values into a 1D array
    #predicted_ffts = collapse(predicted)
    predicted_ffts = np.reshape(predicted, (-1,))  # concatenate all the fft values together
    print 'predicted_ffts', predicted_ffts.shape, predicted_ffts.dtype
   
    # Get rid of the trailing zeros 
    predicted_ffts = np.trim_zeros(predicted_ffts, 'b')
    print 'predicted_ffts trimmed', predicted_ffts.shape, predicted_ffts.dtype
    # Do the inverse FFT to the predicted data, and only consider its real values for playback
    inverted_wav_data = ifft(predicted_ffts)
    print 'inverted_wav_data', inverted_wav_data.shape, inverted_wav_data.dtype

    inverted_wav_data = inverted_wav_data.real

    print 'inverted_wav_data real', inverted_wav_data.shape, inverted_wav_data.dtype
    sd.play(inverted_wav_data, 16000.0)
    inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

    print 'inverted_wav_data squeezed', inverted_wav_data.shape, inverted_wav_data.dtype
    # Scales the waveform to be between -1 and 1
    maxVec = np.max(inverted_wav_data)
    minVec = np.min(inverted_wav_data)
    inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2
    print 'inverted_wav_data to write', inverted_wav_data.shape, inverted_wav_data.dtype
    wav.write(filename + '.wav', 16000.0, inverted_wav_data)


def autoencoder(max_num_frames, num_samples_per_frame, state_size_1, state_size_2, version):
    input_placeholder = tf.placeholder(tf.complex64, (None, max_num_frames, num_samples_per_frame))
    real = complex_to_float_tensor(input_placeholder)
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(real) * corrupt_prob + real * (1 - corrupt_prob)

    encoder, output = encode(current_input, max_num_frames, num_samples_per_frame, state_size_1, state_size_2, version)
    decoded = decode(encoder, output, max_num_frames, num_samples_per_frame, state_size_1, state_size_2)
    
    loss = tf.losses.mean_squared_error(decoded, real) 
    loss = tf.reduce_mean(loss)

    return {'input': input_placeholder, 'output': output, 'decoded': decoded, 'corrupt_prob': corrupt_prob, 'cost': loss, 'encoder': encoder}


def preprocess_data(num_samples_per_frame, window_len, window_step, max_num_frames):
    """Processes the training data and returns MFCC vectors for all of them.
    Args:
    Returns:
    train_data: A list, one for each training example: accent 1 padded FFT data 
    train_labels: A list, one for each training example: accent 2 padded FFT data
    """
    inputs = [] 
    labels = [] 

    SOURCE_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
    TARGET_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'    
    index = 0
    for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
        if index >= 200:
            break
        index += 1
        print source_fname

        if source_fname == '.DS_Store' or target_fname == '.DS_Store':
            continue

        (source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
        (target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

        src_fft = fft(source_wav_data)  # both of these are complex numbers
        tgt_fft = fft(target_wav_data)  

        # pads the MFCC feature matrices (rows) to length max_num_frames
        source_padded_frames = pad_sequence(src_fft, max_num_frames, num_samples_per_frame=num_samples_per_frame)
        target_padded_frames = pad_sequence(tgt_fft, max_num_frames, num_samples_per_frame=num_samples_per_frame)

        source_padded_frames = np.reshape(source_padded_frames, (max_num_frames, num_samples_per_frame)) 
        target_padded_frames = np.reshape(target_padded_frames, (max_num_frames, num_samples_per_frame))

        inputs.append(source_padded_frames) 
        labels.append(target_padded_frames)
        print 'data shape', source_padded_frames.shape, source_padded_frames.dtype 

    return inputs, labels


def autoencode():
    batch_size = 5
    lr = 1e-3
    momentum = 0.3
   
    sample_rate = 16000.0
    window_len = 0.005   # 5 ms
    window_step = 0.005  # 5 ms 

    max_num_samples = 85120
    num_samples_per_frame = int(sample_rate * window_len)  # = 80 samples/frame 
    max_num_frames = max_num_samples / num_samples_per_frame     # = 1064 because the max recording has 85120 samples, and 85120 / 80 = 1064 

    num_features = max_num_frames * num_samples_per_frame 
    state_size_1 = 50 
    state_size_2 = 50 
    epochs = 30

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    accent_a, accent_b = preprocess_data(num_samples_per_frame, window_len, window_step, max_num_frames)
    ae1 = autoencoder(max_num_frames, num_samples_per_frame, state_size_1, state_size_2, '1')
    ae2 = autoencoder(max_num_frames, num_samples_per_frame, state_size_1, state_size_2, '2')

    optimizer1 = tf.train.AdamOptimizer(lr).minimize(ae1['cost'])
    optimizer2 = tf.train.AdamOptimizer(lr).minimize(ae2['cost'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # learn weights for the first accent
    print 'TRAINING ON ACCENT A'
    for epoch_i in range(epochs):
        n_minibatches, total_loss = 0, 0
        minibatches, _ = get_minibatches(accent_a, accent_b, batch_size)
        for batch in minibatches:
            sess.run(optimizer1, feed_dict={ae1['input']: batch, ae1['corrupt_prob']: [1.0]})
            total_loss += ae1['cost'].eval(session=sess, feed_dict={ae1['input']: batch, ae1['corrupt_prob']: [1.0]})
            decoded = ae1['decoded'].eval(session=sess, feed_dict={ae1['input']: batch, ae1['corrupt_prob']: [1.0]})

            n_minibatches += 1
        print epoch_i, total_loss/n_minibatches

    encoder = ae1['encoder']

    # learn weights for the second accent
    print 'TRAINING ON ACCENT B'
    for epoch_i in range(epochs):
        n_minibatches, total_loss = 0, 0
        minibatches, _ = get_minibatches(accent_b, accent_a, batch_size)
        for batch in minibatches:
            sess.run(optimizer2, feed_dict={ae2['input']: batch, ae2['corrupt_prob']: [1.0]})
            total_loss += ae2['cost'].eval(session=sess, feed_dict={ae2['input']: batch, ae2['corrupt_prob']: [1.0]})
            decoded = ae2['decoded'].eval(session=sess, feed_dict={ae2['input']: batch, ae2['corrupt_prob']: [1.0]})

            n_minibatches += 1
        print epoch_i, total_loss/n_minibatches
    decoder = ae2['encoder']


    print 'EVALUATING CONVERSION FROM A TO B'
    minibatches_a, minibatches_b = get_minibatches(accent_a, accent_b, batch_size)
    for i in range(0, len(minibatches_a)):
        raw_batch = minibatches_a[i]
        target_batch = minibatches_b[i]
        batch = complex_to_float_numpy(raw_batch)
        encode_W1 = encoder[1].eval(session=sess)
        encode_W2 = encoder[0].eval(session=sess)
        decode_W1 = np.transpose(decoder[0].eval(session=sess))
        decode_W2 = np.transpose(decoder[1].eval(session=sess))
        h1 = np.tanh(batch_multiply_by_matrix_numpy(np.array(batch), encode_W1))
        h2 = np.tanh(batch_multiply_by_matrix_numpy(h1, encode_W2))
        h3 = np.tanh(batch_multiply_by_matrix_numpy(h2, decode_W1))
        prediction = batch_multiply_by_matrix_numpy(h3, decode_W2)
        fft_preds_reals = tf.slice(prediction, [0, 0, 0], [-1, max_num_frames, num_samples_per_frame]) 
        fft_preds_complexes = tf.slice(prediction, [0, 0, num_samples_per_frame], [-1, max_num_frames, num_samples_per_frame])
        pred = tf.complex(fft_preds_reals, fft_preds_complexes).eval(session=sess)
        output_wave_file(pred[0], filename='autoencoder_pred_'+str(i))   
        output_wave_file(target_batch[0], filename='autoencoder_input_'+str(i))

autoencode()











