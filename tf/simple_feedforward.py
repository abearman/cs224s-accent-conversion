from time import time, gmtime, strftime
import numpy as np
import os
import random
import matlab.engine

import scipy.io.wavfile as wav
from python_speech_features import mfcc

import tensorflow as tf

from utils.general_utils import get_minibatches, batch_multiply_by_matrix
from utils.fast_dtw import get_dtw_series 
from utils.pad_sequence import pad_sequence


def preprocess_data(num_mfcc_coeffs, max_num_frames):
    """Processes the training data and returns MFCC vectors for all of them.
    Args:
        num_mfcc_coeffs, max_num_frames
    Returns:
        train_data: A list of tuples, one for each training example: (accent 1 padded MFCC frames, accent 1 mask)
        train_labels: A list of tuples, one for each training example: (accent 2 padded MFCC frames, accent 2 mask)
    """
    inputs = [] 
    labels = [] 
    input_masks = []
    label_masks = []

    #SOURCE_DIR = '../data/cmu_arctic/us-english-male-bdl/wav/'
    #TARGET_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'
    SOURCE_DIR = '../data/cmu_arctic/mini_a/'
    TARGET_DIR = '../data/cmu_arctic/mini_b/'
    for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
        (source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
        (target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

        source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=num_mfcc_coeffs))
        target_mfcc_features = np.array(mfcc(target_wav_data, samplerate=target_sample_rate, numcep=num_mfcc_coeffs))

        # Aligns the MFCC features matrices using FastDTW.
        source_mfcc_features, target_mfcc_features = get_dtw_series(source_mfcc_features, target_mfcc_features)

        # Pads the MFCC feature matrices (rows) to length config.max_num_frames
        source_padded_frames, source_mask = pad_sequence(source_mfcc_features, max_num_frames)
        target_padded_frames, target_mask = pad_sequence(target_mfcc_features, max_num_frames)

        inputs.append(source_padded_frames) 
        input_masks.append(source_mask)
        labels.append(target_padded_frames) 
        label_masks.append(target_mask) 

    randomized_indices = range(0, len(inputs)) 
    random.shuffle(randomized_indices)
    inputs = [inputs[i] for i in randomized_indices]
    input_masks = [input_masks[i] for i in randomized_indices]
    labels = [labels[i] for i in randomized_indices]
    label_masks = [label_masks[i] for i in randomized_indices] 

    return inputs, labels


def forward_prop(W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, inputs, num_frames, num_mfcc):
    h1 = tf.nn.relu(batch_multiply_by_matrix(matrix=W1, batch=inputs) + b1)
    print "h1 shape: ", h1

    # [batch, state_size1] x [state_size1, state_size2] = [batch, state_size2]
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    print "h2 shape: ", h2

    # [batch, state_size2] x [state_size2, state_size3] = [batch, state_size3]
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    print "h3 shape: ", h3

    # [batch, state_size3] x [state_size3, max_num_frames * num_mfcc_coeffs] = [batch, max_num_frames, num_mfcc_coeffs]
    print "W4 shape: ", W4
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)

    mfcc_preds = tf.nn.relu(tf.matmul(h4, W5) + b5)
    mfcc_preds = tf.reshape(mfcc_preds, (-1, num_frames, num_mfcc))
    print "mfcc preds shape: ", mfcc_preds

    return mfcc_preds 


def main():
    print "Starting matlab ... type in your password if prompted"
    eng = matlab.engine.start_matlab()
    eng.addpath('../invMFCCs_new')
    print "Done starting matlab"

    batch_size = 32 
    n_epochs = 50
    lr = 1e-3
    max_num_frames = 706 
    num_mfcc_coeffs = 25
    sample_rate = 16000.0
    num_features = max_num_frames * num_mfcc_coeffs 
    state_size_1 = 25
    state_size_2 = 50
    state_size_3 = 50
    state_size_4 = 25

    inputs, labels = preprocess_data(num_mfcc_coeffs, max_num_frames)
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


    input_placeholder = tf.placeholder(tf.float32, (None, max_num_frames, num_mfcc_coeffs))
    labels_placeholder = tf.placeholder(tf.float32, (None, max_num_frames, num_mfcc_coeffs))

    xavier = tf.contrib.layers.xavier_initializer()
    W1 = tf.get_variable("W1", shape=(num_features, state_size_1), initializer=xavier) 
    b1 = tf.get_variable("b1", shape=(1, state_size_1))

    W2 = tf.get_variable("W2", shape=(state_size_1, state_size_2), initializer=xavier) 
    b2 = tf.get_variable("b2", shape=(1, state_size_2))

    W3 = tf.get_variable("W3", shape=(state_size_2, state_size_3), initializer=xavier) 
    b3 = tf.get_variable("b3", shape=(1, state_size_3))

    W4 = tf.get_variable("W4", shape=(state_size_3, state_size_4), initializer=xavier) 
    b4 = tf.get_variable("b4", shape=(1, state_size_4))

    W5 = tf.get_variable("W5", shape=(state_size_4, num_features), initializer=xavier) 
    b5 = tf.get_variable("b5", shape=(1, num_features))

    # forward propagation
    mfcc_preds = forward_prop(W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, input_placeholder, max_num_frames, num_mfcc_coeffs)

    # backward propagation
    loss = tf.reduce_mean(tf.squared_difference(mfcc_preds, labels_placeholder))
    updates = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter(logs_path + '/train', sess.graph)

    for epoch in range(n_epochs):
        start_time = time()
        # train with each batch
        n_minibatches = 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], batch_size):
            n_minibatches += 1
            feed = {input_placeholder: input_batch, labels_placeholder: labels_batch}
            sess.run(updates, feed_dict=feed)

            #train_writer.add_summary(summary, epoch_count)

        duration = time() - start_time
        print 'Epoch ' + str(epoch) + ' : loss = ' + str(loss.eval(session=sess, feed_dict=feed)) + ' (' + str(duration) + ' sec)'

        predicted_mfccs_batch = mfcc_preds.eval(session=sess, feed_dict=feed)
        for i in range(predicted_mfccs_batch.shape[0]):
            predicted_mfccs_transposed = np.transpose(predicted_mfccs_batch[i,:,:])

            # MFCC features need to be a numpy array of shape (num_coefficients x num_frames) in order to be passed to the invmelfcc function
            inverted_wav_data = eng.invmelfcc(matlab.double(predicted_mfccs_transposed.tolist()), sample_rate, num_mfcc_coeffs)

           # eng.soundsc(inverted_wav_data, sample_rate, nargout=0)
            inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

            # Scales the waveform to be between -1 and 1
            maxVec = np.max(inverted_wav_data)
            minVec = np.min(inverted_wav_data)
            inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2

            wav.write('learned_wav' + str(i) + '.wav', sample_rate, inverted_wav_data)

    sess.close()

if __name__ == '__main__':
    main()