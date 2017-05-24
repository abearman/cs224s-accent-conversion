import os
#from pymatbridge import Matlab
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matlab.engine
import numpy as np
from sklearn.preprocessing import normalize
import librosa


SOURCE_DIR = 'data/cmu_arctic/us-english-male-bdl/wav/'
TARGET_DIR = 'data/cmu_arctic/scottish-english-male-awb/wav/'
for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
	(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
	(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

	print "Starting matlab ... type in your password if prompted"
	eng = matlab.engine.start_matlab()
	eng.addpath('invMFCCs_new')
	print "Finished starting matlab"

	# MFCC features need to be a numpy array of shape (num_coefficients x num_frames) in order to be passed to the invmelfcc function
	source_mfcc_features1 = np.array(eng.melfcc(matlab.double(source_wav_data.tolist()), float(source_sample_rate)))	
	source_mfcc_features = np.transpose(np.array(mfcc(source_wav_data, source_sample_rate)))
	print "src mfcc features shape: ", source_mfcc_features.shape
	print "src features matlab: ", source_mfcc_features1
	print "src features python: ", source_mfcc_features

	# The reconstructed waveform
	inverted_wav_data = eng.invmelfcc(matlab.double(source_mfcc_features.tolist()), float(source_sample_rate)) 
	eng.soundsc(inverted_wav_data, float(source_sample_rate), nargout=0)
	inverted_wav_data = np.squeeze(np.array(inverted_wav_data))

	# Scales the waveform to be between -1 and 1
	maxVec = np.max(inverted_wav_data)
	minVec = np.min(inverted_wav_data)
	inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2
	
	wav.write('source.wav', source_sample_rate, source_wav_data)
	wav.write('predicted_target.wav', source_sample_rate, inverted_wav_data) 

 	exit()
