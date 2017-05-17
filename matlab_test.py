import os
#from pymatbridge import Matlab
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matlab.engine
import numpy as np


#mlab = Matlab(matlab='/Applications/MATLAB_R2017a.app/bin/matlab')
#mlab.start()

SOURCE_DIR = 'data/cmu_arctic/us-english-male-bdl/wav/'
TARGET_DIR = 'data/cmu_arctic/scottish-english-male-awb/wav/'
for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):
	(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
	(target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)
	source_mfcc_features = mfcc(source_wav_data, source_sample_rate)	 # Returns a numpy array of num_frames x num_cepstrals
	target_mfcc_features = mfcc(target_wav_data, target_sample_rate)

	print "Starting matlab ... type in your password if prompted"
	eng = matlab.engine.start_matlab()
	eng.addpath('matlab_2')
	
	inverted = eng.invmelfcc(matlab.double(source_mfcc_features.tolist()))
	eng.soundsc(inverted, 16000.0, nargout=0)

 	exit()