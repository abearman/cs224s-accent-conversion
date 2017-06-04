import numpy as np
import os
import matlab.engine
import scipy.io.wavfile as wav
from python_speech_features import mfcc


batch_size = 5
max_num_frames = 1220  # max length of warped time series
num_mfcc_coeffs = 25
sample_rate = 16000.0
num_filters = 100
window_len = 0.005   # 5 ms
window_step = 0.005  # 5 ms 
num_features = max_num_frames * num_mfcc_coeffs 


print "Starting matlab ... type in your password if prompted"
eng = matlab.engine.start_matlab()
eng.addpath('../invMFCCs_new')
print "Done starting matlab"

SOURCE_DIR = '../data/cmu_arctic/scottish-english-male-awb/wav/'
DEST_DIR = '../data/cmu_arctic/scottish-english-male-awb/reconstructed_wav/'
for source_fname in os.listdir(SOURCE_DIR):
	print source_fname
	if source_fname == '.DS_Store':
		continue
	(source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
	source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=num_mfcc_coeffs, nfilt=num_filters, winlen=window_len, winstep=window_step))

	transposed = np.transpose(source_mfcc_features)
	inverted_wav_data = eng.invmelfcc(matlab.double(transposed.tolist()), 16000.0, 25, 100.0, 0.005, 0.005)
	maxVec = np.max(inverted_wav_data)
	minVec = np.min(inverted_wav_data)
	inverted_wav_data = ((inverted_wav_data - minVec) / (maxVec - minVec) - 0.5) * 2

	wav.write(DEST_DIR + source_fname + '.wav', 16000.0, inverted_wav_data)