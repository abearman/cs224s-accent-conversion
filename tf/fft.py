import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import blackmanharris  


def freq_from_FFT(sig, fs):
	# Compute Fourier transform of windowed signal
	N = len(sig)
	print "N: ", N
	print "fs: ", fs
	windowed = sig * blackmanharris(N)
	X = np.abs(np.fft.rfft(windowed))
	print X
	print X.shape	

	exit()

	# Find the peak and interpolate
	i = np.argmax(abs(X)) # Just use this for less-accurate, naive version
	X[X == 0] = epsilon		# Circumvent division by 0

	true_i = interpolate(X, i)[0]
	
	return fs * true_i / N


filename = '../data/cmu_arctic/us-english-male-bdl/wav/arctic_b0539.wav'
fs, wav_data = wav.read(filename)
#np.fft.fft(wav_data)
print freq_from_FFT(wav_data, fs)
