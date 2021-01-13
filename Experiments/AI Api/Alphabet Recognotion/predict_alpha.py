# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:19:06 2018

@author: User
"""
import cv2
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
from keras.models import model_from_json
K.set_image_dim_ordering('th')

def modify(img, threshold=0.5):
    for row in range(28):
        for col in range(28):
            if ((img[0][row][col] < threshold) and (row>3 and row < 24) and (col>3 and col < 24)):
                img[0][row][col] = 1
            else:
                img[0][row][col] = 0

def display(img, threshold=0.5):
    # Debugging only
#    render = ''
#    for row in range(28):
#        for col in range(28):
#            if img[0][row][col] > threshold:
#                render += '@'
#            else:
#                render += '.'
#        render += '\n'
#    print(render)
    
    image = cv2.imread('testImg.png')
    image = cv2.resize(image, (28, 28))
    
    for row in range(28):
        for col in range(28):
            if img[0][row][col] > threshold:
                image[row][col]=1
            else:
                image[row][col]=0
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    plt.show()
    
'''
dataset = numpy.loadtxt("Name_Test_6.txt", delimiter=",")
dim = 28*28
X = dataset[:,0:dim]
#y = dataset[:,dim]

X = X.reshape(X.shape[0] , 1 , 28 , 28).astype('float32')

for i  in range(X.shape[0]):
    modify(X[i])

for i  in range(X.shape[0]):
#    print(" ====> " , int(y[i]))
#    print(i,X[i].shape,int(y[i]))
    display(X[i])
'''

json_file = open('model_alpha_coopes.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_alpha_coopes.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dim = 28*28
test_data = numpy.loadtxt("alpha_testing_dataset.txt", delimiter=",")
X_test = test_data[:,0:dim]
y_test = test_data[:,dim]
X_test = X_test.reshape(X_test.shape[0] , 1 , 28 , 28).astype('float32')
yy = y_test
y_test = np_utils.to_categorical(y_test)

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

num_classes = 26
prediction = model.predict(X_test)
for i in range(20):
    now = -1
    mx = -1
    for j in range(num_classes):
        if(prediction[i][j] > mx): 
            now = j;
            mx = prediction[i][j]
    display(X_test[i])
    print(" ------------------> " , mx , chr(ord('A') + now))
    
    
    