import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import sys
import os
import struct
import tensorflow as tf
from numpy import asarray
from PIL import Image
from scipy import misc
import pathlib
import cv2


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.


    """
    def __init__(self, n_hidden=500,
                 l2=0., epochs=15, eta=0.01,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.activation = 'sigmoid'
        self.w_h = self.random.normal(loc=0.0, scale=0.1,size=(784,self.n_hidden))

        self.w_h2 = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_hidden))

        # weights for hidden -> output

        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden,
                                              10))

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation
        """
        onehot = np.full((n_classes, y.shape[0]), 0.01)
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = .99
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_examples, n_features] dot [n_features, n_hidden]
        # -> [n_examples, n_hidden]
        zh1 = np.dot(X, self.w_h)

        # step 2: activation of hidden layer
        ah1 = self._sigmoid(zh1)

        zh2 = np.dot(ah1, self.w_h2)
        ah2 = self._sigmoid(zh2)

        # step 3: net input of output layer
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]

        z_out = np.dot(ah2, self.w_out)
        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return zh1, ah1, zh2, ah2, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_examples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        error =np.sum((y_enc - output)**2)
        error = error/len(output)
        
##        L2_term = (self.l2 *
##                   (np.sum(self.w_h ** 2.) +
##                    np.sum(self.w_out ** 2.)))
##
##        term1 = -y_enc * (np.log(output))
##        term2 = (1. - y_enc) * np.log(1. - output)
##        cost = np.sum(term1 - term2) + L2_term
        return error

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            Predicted class labels.

        """
        z_h, a_h, zh2, ah2, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0] # no. of class
                                               #labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden

        

        epoch_strlen = len(str(self.epochs)) # for progr. format.
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': \
                      []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] -\
                                   self.minibatch_size +\
                                   1, self.minibatch_size):
                

                batch_idx = indices[start_idx:start_idx +\
                                    self.minibatch_size]
               

                # forward propagation
                z_h, a_h, zh2, ah2, z_out, a_out = \
                    self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
                sigmoid_derivative_h2 = ah2 * (1. - ah2)

                # [n_examples, n_classlabels] dot [n_classlabels,
                #                                 n_hidden]
                # -> [n_examples, n_hidden]
                delta_h2 = np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h2
                delta_h = (np.dot(delta_h2, self.w_h2.T) *
                           sigmoid_derivative_h)

                # [n_features, n_examples] dot [n_examples,
                #                               n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_w_h2 = np.dot(ah2.T, delta_h2)

                # [n_hidden, n_examples] dot [n_examples,
                #                            n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)


                # Regularization and weight updates
                delta_w_h = grad_w_h + self.l2*self.w_h
                delta_w_h2 = (grad_w_h2 + self.l2*self.w_h2)

                self.w_h -= self.eta * delta_w_h
                self.w_h2 -= self.eta * delta_w_h2

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                # bias is not regularized
                self.w_out -= self.eta * delta_w_out


            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h,zh2, ah2, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train ==
                          y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid ==
                          y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% '
                              %
                             (epoch_strlen, i+1, self.epochs,
                              cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

    def query(self, X, Y):
        acc = 0
        for x, y in zip(X, Y):
            s = self.predict(x)
            
            if s[0] == y:
                acc +=1
        return acc/len(X) * 100

def main():

    mnist = np.load('mnist_scaled.npz')
    X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
    
    #hyper parameters to be tested
    hiddenLayers =[100, 200, 300,400, 500, 600]
    LR = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6]
    epochs = [1, 5, 10, 15, 20]
    costs = []
    
    
    plt.ylabel('Cost')
    plt.xlabel('Epochs')

    #initial training 
    nn = NeuralNetMLP(epochs =20, n_hidden =600, eta = 0.01)
    nn.fit(X_train, y_train, X_test, y_test)
    costs.append(nn.eval_['cost'])
    plt.plot(range(nn.epochs), nn.eval_['cost'], color = 'purple')

    #TESTING WITH MY OWN IMAGES AND THE GOOD HYPER PARAMETERS
##    imageName = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
##    labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
##    images = []
##    for name in imageName:
##        
##        image = cv2.imread(name) 
##        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
##        image = np.expand_dims(image, 2)
##        numpydata = asarray(image)
##        numpydata = np.resize(numpydata, (28,28,1))
##        numpydata = numpydata.reshape(numpydata.shape[2], 784)
##        numpydata = numpydata.astype('float32')
##        numpydata = numpydata/255
##        images.append(numpydata)
##    images = asarray(images)
    #testing my model with my images 
    #nn = NeuralNetMLP(n_hidden =500, epochs = 15, eta = 0.01)
    #nn.fit(X_train, y_train, X_test, y_test)
    #prediction = nn.query(images, np.array(labels))
    #print("how accurate it was with my images", prediction)
    
    #costs.append(nn.eval_['cost'])
    #plt.plot(range(nn.epochs), nn.eval_['cost'], color = 'blue')

    #TRAINING WITH ROTATED IMAGES AND ORIGINAL
    
    #X_train = X_train.reshape((28,28))
 
    
    
##        
##    X_train_rotate_pos = np.apply_along_axis(rotatePos, axis = 1, arr = X_train)
##    X_train_rotate_neg = np.apply_along_axis(rotateNeg, axis = 1, arr = X_train)
##    X_train_rotate_pos = X_train_rotate_pos.reshape((60000, 784))
##    X_train_rotate_neg = X_train_rotate_neg.reshape((60000, 784))
##    
##
##
##    X_train = np.vstack((X_train,X_train_rotate_pos))
##    X_train = np.vstack((X_train,X_train_rotate_neg ))
##    y_train2 = np.concatenate((y_train, y_train))
##    y_train2 = np.concatenate((y_train2, y_train))
##    nn = NeuralNetMLP(n_hidden =500, epochs = 20, eta = 0.01)
##    nn.fit(X_train, y_train2, X_test, y_test)
##    costs.append(nn.eval_['cost'])
##    plt.plot(range(nn.epochs), nn.eval_['cost'], color = 'blue')
##    plt.show()
    

    
    #cost = nn.query(lyst)
    #print(cost)

## USED FOR TUNING HYPER PARAMETERS
    colors = ['red','blue', 'green', 'purple', 'yellow', 'black' ]
##    for i in range(len(hiddenLayers)):
##        
##        nn = NeuralNetMLP(n_hidden = hiddenLayers[i], epochs = 5, eta =0.001)
##        nn.fit(X_train, y_train, X_test, y_test)
##        #costs.append(nn.eval_['cost'])
##        
##        plt.plot(range(nn.epochs), nn.eval_['cost'], color = colors[i], label = hiddenLayers[i])
##
##    plt.legend()
##    plt.show()
##
##    for i in range(len(LR)):
##        
##        nn = NeuralNetMLP(eta = LR[i], epochs = 5, n_hidden = 200)
##        nn.fit(X_train, y_train, X_test, y_test)
##        
##        
##        plt.plot(range(nn.epochs), nn.eval_['cost'], color = colors[i], label = LR[i])
##    plt.legend()
##    plt.show()

    for i in range(len(epochs)):
        
        nn = NeuralNetMLP(epochs = epochs[i], n_hidden=5, eta =0.01)
        nn.fit(X_train, y_train, X_test, y_test)
       
        
        plt.plot(range(nn.epochs), nn.eval_['cost'], color = colors[i], label = epochs[i])
    plt.legend()
    plt.show()
def rotatePos(row):
    row = row.reshape((28,28))
    rotated = scipy.ndimage.interpolation.rotate(row, angle = 10, reshape = False)
    rotated = rotated.reshape((1, 784))
    return rotated

def rotateNeg(row):
    row = row.reshape((28,28))
    rotated = scipy.ndimage.interpolation.rotate(row, angle = -10, reshape = False)
    rotated = rotated.reshape((1, 784))
    return rotated


def load_mnist(path, kind='train'):
    """Load MNIST data from 'path'"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                             len(labels), 784)
        images = ((images / 255.) - .5) * 2
    
    return images, labels
    
         


if __name__ =="__main__":
    main()
