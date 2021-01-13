# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:50:05 2018

@author: User
"""

from scipy.io import loadmat
import pickle
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy
K.set_image_dim_ordering('th')

train_file = open('alpha_training_dataset_sample.txt','w')
test_file = open('alpha_testing_dataset_sample.txt','w')

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    max_ = 24
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def reshape(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        img.shape = (width,height)
        img = img.T
        img = list(img)
        img = [item for sublist in img for item in sublist]
        return img

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        print(render)
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]-1:kv[1:][0] for kv in mat['dataset'][0][0][2]}
#    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_]
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_] #shift matlab indicies to start from 0

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_]
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_] #shift matlab indicies to start from 0

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = reshape(training_images[i])
    if verbose == True: print('')
    
    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = reshape(testing_images[i])
    if verbose == True: print('')

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0],1, height, width)
    testing_images = testing_images.reshape(testing_images.shape[0],1, height, width)
    
    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)
    print(nb_classes)
    return ((training_images, training_labels), (testing_images, testing_labels), nb_classes)

(X_train,y_train),(X_test,y_test),nb_classes = load_data("E:\Api Codes\EMNIST dataset\matlab\emnist-letters")

def saveData(img, threshold , flag):
    # Debugging only
#    render = ''
    for row in range(28):
        for col in range(28):
            if img[0][row][col] > threshold:
#                render += '@'
                val = 1
            else:
#                render += '.'
                val = 0
            if(flag == 0):
                train_file.write(str(val) + ', ')
            else:
                test_file.write(str(val) + ', ')
            
#        render += '\n'
#    print(render)

print(X_train.shape , y_train.shape , X_test.shape , y_test.shape)

print(' ---> saving training data')
for i in range(len(X_train)):
#    plt.subplot(331+(i%9))
#    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#    print(" ====> " , int(y_train[i][0] - 1))
#    print(i,X_train[i].shape,chr(ord('A') + y_train[i][0] - 1))
    saveData(X_train[i] , 0.5 , 0)
    train_file.write(str(y_train[i][0] - 1) + '\n')

print(' ---> saving testing data')
for i in range(len(X_test)):
#    plt.subplot(331+(i%9))
#    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#    print(i,X_test[i].shape,chr(ord('A') + y_test[i][0] - 1))
    saveData(X_test[i] , 0.5 , 1)
    test_file.write(str(y_test[i][0] - 1) + '\n')
    
train_file.close()
test_file.close()
print("Loading finished")














