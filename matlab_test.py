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
	source_mfcc_features_np = mfcc(source_wav_data, source_sample_rate)	 # Returns a numpy array of num_frames x num_cepstrals
	target_mfcc_features = mfcc(target_wav_data, target_sample_rate)

	
	source_mfcc_features = list()
	for array in source_mfcc_features_np:
		source_mfcc_features.append(list())
		for number in array:
			source_mfcc_features[-1].append(matlab.double(number))
		# source_mfcc_features.append(matlab.double(array.tolist()))

	print source_mfcc_features

	# print source_mfcc_features.tolist()

	print "Starting matlab ... type in your password if prompted"
	eng = matlab.engine.start_matlab()
	eng.addpath('matlab_2')
	#print "Inverting MFCCs for wav file ", source_fname
	#res = mlab.run('matlab_2/invmelff', {'cep': source_mfcc_features.tolist(), 'sr': source_sample_rate})
	
	inverted = eng.invmelfcc(matlab.double(source_mfcc_features), matlab.double(source_sample_rate))
	eng.soundsc(res['result'], source_sample_rate)
 	#eng.addpath('../invMFCCs')
	#print "Running invMFCCs on wav file: ", source_fname
	#eng.invMFCCs(SOURCE_DIR + source_fname, nargout=0)  # Need nargout=0 if there are no values returned 

 	exit()