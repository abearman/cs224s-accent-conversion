import os
import numpy
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


"""(rate,sig) = wav.read('data/cmu_arctic/canadian-english-male-jmk/wav/arctic_a0001.wav')
canadian = mfcc(sig,rate)

(rate,sig) = wav.read('data/cmu_arctic/us-english-male-bdl/wav/arctic_a0001.wav')
american1 = mfcc(sig,rate)

(rate,sig) = wav.read('data/cmu_arctic/us-english-male-bdl/wav/arctic_a0002.wav')
american2 = mfcc(sig,rate)

data = []
labels = []
for datapoint in canadian:
	data.append(datapoint)
	labels.append(1)

for datapoint in american1:
	data.append(datapoint)
	labels.append(0)

test_data = []
for datapoint in american2:
	test_data.append(datapoint)

model = svm.SVC()
model.fit(data, labels)
print(model.predict(test_data))"""

def classify():
	data = []
	labels = []
	test_data = []
	test_labels = []
	test_names = ['arctic_b0535.wav', 'arctic_b0536.wav', 'arctic_b0537.wav', 'arctic_b0538.wav', 'arctic_b0539.wav']

	for fname in os.listdir('data/cmu_arctic/canadian-english-male-jmk/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/canadian-english-male-jmk/wav/'+fname)
		sample = mfcc(sig,rate)
		print fname
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(0) # 0 = canadian
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(0) # 0 = canadian

	for fname in os.listdir('data/cmu_arctic/us-english-male-bdl/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/us-english-male-bdl/wav/'+fname)
		sample = mfcc(sig,rate)
		print fname
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(1) # 1 = american
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(1) # 1 = american


	model = GaussianNB()
	model.fit(data, labels)
	print 'model fit'
	pred = model.predict(test_data)
	print(classification_report(pred, test_labels))

classify()