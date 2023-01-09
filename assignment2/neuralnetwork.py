"""
Author:Ana Estrada
File: neuralnetwork.py

activation function: relu
optomization alg: stochastic gradient descent
learning rate: 0.01
max iter = 50
verbose = 10
random state =1
alpha = 1e-5
"""

# Standard scientific Python imports
from matplotlib import pyplot

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import keras
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(y_test)

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255
X_test = X_test/255


classifier = MLPClassifier(hidden_layer_sizes = (100,100), activation= 'relu', solver = 'sgd', alpha = 1e-5,
              learning_rate_init = 0.01, max_iter = 50, verbose = 10,
              random_state = 1)

classifier = classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_predict)

disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("confusion matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()

loss_values = classifier.loss_curve_

print("Training set score: %f" % classifier.score(X_train, y_train))
print("Test set score: %f" % classifier.score(X_test, y_test))

plt.plot(loss_values)
plt.ylabel("loss")
plt.xlabel("iterations")
plt.show()


