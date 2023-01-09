# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

from keras.models import Sequential
from keras.layers import Dense 
from tensorflow.keras.layers import BatchNormalization
# Common imports
import numpy as np
import os

#model summary
print("This learning model aims at classifying different articles of clothing based on pictures."
        "It takes in pictures as the explanatory variable and the response variable is the classification of clothing."
      "It uses an elu activation function, normaliizaes the weights using a normal distribution center on 0."
      "The model uses a SDG algorithm with learning rate of 0.01, momentum of 0.8 and it uses gradient clipping of 0.5."
      "The model usese two hidden desnse layers that have 300 nodes each and then finished with  softmax output layer that has ten nodes and uses a batch normailzation inbetween every layer."
      "The data is pre-flattened. It uses sparse categorical loss and the loss fucntion. The data is an 80:20 split for training:testing.")

# to make this notebook's output stable across runs
np.random.seed(42)
    
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

tf.random.set_seed(42)
np.random.seed(42)

X_train_full = X_train_full.reshape(X_train_full.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)


X_train_full = X_train_full.astype('float32')
X_test = X_test.astype('float32')

X_train_full = X_train_full/255
X_test = X_test/255




#gradient clipping

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum =0.9, clipvalue = 1)

model = keras.Sequential(layers=[
    keras.layers.Dense(300, activation ='elu', kernel_initializer = keras.initializers.he_normal()),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(100, activation="elu", kernel_initializer = keras.initializers.he_normal()),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(10, activation="softmax", kernel_initializer = keras.initializers.he_normal())
])

model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_full, y_train_full, batch_size=1, epochs=10, validation_split=0.2)
loss, accuracy =model.evaluate(X_test, y_test)
print("Accuracy using the testing data", accuracy)
print("loss using the testing data", loss)
