# Gustavo L. Mourao

import numpy as np

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics

# Storage model
from sklearn.externals import joblib

# Neural network library
from sklearn.neural_network import MLPClassifier

# Nearest Centroid Classifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

# KNN
from sklearn.neighbors import KNeighborsClassifier

# -------- Select classifier type --------#
# classifierType = raw_input("Which classifier: (SVM/ANN/KNN/NearestCentroid):")
# classifierType
classifierType = 'SVM'

if classifierType == 'SVM':
	# Create a classifier: a support vector classifier
	classifier = svm.SVC(gamma=0.001,kernel='linear')
elif classifierType == 'NearestCentroid':
	# Nearest Centroid Classifier
	classifier = NearestCentroid()
elif classifierType == 'KNN':
	# KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors=23)
elif classifierType == 'ANN':
	# Neural network
	mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
	                    solver='sgd', verbose=10, tol=1e-4, random_state=1)
	classifier = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=100, alpha=1e-4,
	                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
	                    learning_rate_init=.1)
else:
	print("Possible options: SVM/ANN/KNN/NearestCentroid")
	exit()

# ----------------- Get set of training data collected from LeapMotion sensor ----------------- #
data = np.loadtxt("TrainingInput.txt")
RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0 = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9], data[:,10], data[:,11], data[:,12], data[:,13], data[:,14], data[:,15]

InputSamples = np.vstack((RTP_0,RTP_1,RTP_2,RTP_3,RTP_4,RTT_01,RTT_02,RTT_03,RTT_04,RTT_12,RTT_13,RTT_14,RTT_23,RTT_24,RTT_34,RTJ_0))
InputSamples = InputSamples.T
print ((InputSamples))

dataTarget = np.loadtxt("TargetTraining.txt")
dataTarget = dataTarget.T
n_samples = len(dataTarget)

classifier.fit(InputSamples, dataTarget)
# ----------------- --------------------------------------------------------------------------- #

					
					
# Storage model
if classifierType == 'SVM':
	joblib.dump(classifier, 'SVM.pkl')
elif classifierType == 'NearestCentroid':
	joblib.dump(classifier, 'NearestCentroid.pkl')
elif classifierType == 'KNN':
	joblib.dump(classifier, 'KNN.pkl')
elif classifierType == 'ANN':
	joblib.dump(classifier, 'ANN.pkl')	
else:
	print("Possible options: SVM/ANN/KNN/NearestCentroid")
	exit()	
	
#########################################
# Predict the value of the set of symbols/gestures on the second half:
dataTest = np.loadtxt("TestInput.txt")
RTP_0, RTP_1, RTP_2, RTP_3, RTP_4, RTT_01, RTT_02, RTT_03, RTT_04, RTT_12, RTT_13, RTT_14, RTT_23, RTT_24, RTT_34, RTJ_0 = dataTest[:,0], dataTest[:,1], dataTest[:,2], dataTest[:,3], dataTest[:,4], dataTest[:,5], dataTest[:,6], dataTest[:,7], dataTest[:,8], dataTest[:,9], dataTest[:,10], dataTest[:,11], dataTest[:,12], dataTest[:,13], dataTest[:,14], dataTest[:,15]

InputSamplesTest = np.vstack((RTP_0,RTP_1,RTP_2,RTP_3,RTP_4,RTT_01,RTT_02,RTT_03,RTT_04,RTT_12,RTT_13,RTT_14,RTT_23,RTT_24,RTT_34,RTJ_0))
InputSamplesTest = InputSamplesTest.T

dataTargetTest = np.loadtxt("TestTarget.txt")
dataTargetTest = dataTargetTest.T

expected = dataTargetTest
predicted = classifier.predict(InputSamplesTest)
#########################################

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
conf = metrics.confusion_matrix(expected, predicted)
plt.imshow(conf,cmap='binary',interpolation='None')
plt.show()

