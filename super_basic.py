import os
import numpy
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

def classify():
	data = []
	labels = []
	test_data = []
	test_labels = []
	test_names = ['arctic_b0529.wav','arctic_b0530.wav','arctic_b0531.wav','arctic_b0532.wav','arctic_b0533.wav','arctic_b0534.wav','arctic_b0535.wav','arctic_b0535.wav', 'arctic_b0536.wav', 'arctic_b0537.wav', 'arctic_b0538.wav', 'arctic_b0539.wav']

	print 'CANADIAN'
	for fname in os.listdir('data/cmu_arctic/canadian-english-male-jmk/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/canadian-english-male-jmk/wav/'+fname)
		sample = mfcc(sig,rate)
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(0) # 0 = canadian
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(0) # 0 = canadian


	print 'AMERICAN 1'
	for fname in os.listdir('data/cmu_arctic/us-english-male-bdl/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/us-english-male-bdl/wav/'+fname)
		sample = mfcc(sig,rate)
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(1) # 1 = american
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(1) # 1 = american


	print 'AMERICAN 2'
	for fname in os.listdir('data/cmu_arctic/us-english-male-rms/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/us-english-male-rms/wav/'+fname)
		sample = mfcc(sig,rate)
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(1) # 1 = american
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(1) # 1 = american


	print 'SCOTTISH'
	for fname in os.listdir('data/cmu_arctic/scottish-english-male-awb/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/scottish-english-male-awb/wav/'+fname)
		sample = mfcc(sig,rate)
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(2) # 2 = scottish
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(2) # 2 = scottish


	print 'INDIAN'
	for fname in os.listdir('data/cmu_arctic/indian-english-male-ksp/wav/'):
		(rate,sig) = wav.read('data/cmu_arctic/indian-english-male-ksp/wav/'+fname)
		sample = mfcc(sig,rate)
		if fname in test_names:
			for datapoint in sample:
				test_data.append(datapoint)
				test_labels.append(3) # 3 = indian
		else:
			for datapoint in sample:
				data.append(datapoint)
				labels.append(3) # 3 = indian

	nb_model = GaussianNB()
	nb_model.fit(data, labels)
	print 'Naive Bayes model fit'
	pred = nb_model.predict(test_data)
	print(classification_report(pred, test_labels))
	print(accuracy_score(pred, test_labels))

	print 'SVM model fit'
	svm_model = SVC()
	svm_model.fit(data, labels)
	pred = svm_model.predict(test_data)
	print(classification_report(pred, test_labels))
	print(accuracy_score(pred, test_labels))


classify()
