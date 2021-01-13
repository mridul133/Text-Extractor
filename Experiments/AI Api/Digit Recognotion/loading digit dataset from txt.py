# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:46:35 2018

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
from keras.models import model_from_json
K.set_image_dim_ordering('th')

def display(img, threshold=0.5):
    # Debugging only
    render = ''
    for row in range(28):
        for col in range(28):
            if img[0][row][col] > threshold:
                render += '@'
            else:
                render += '.'
        render += '\n'
    print(render)


train_data = numpy.loadtxt("digit_training_dataset.txt", delimiter=",")
test_data = numpy.loadtxt("digit_testing_dataset.txt", delimiter=",")
print(train_data.shape)

dim = 28*28
X_train = train_data[:,0:dim]
y_train = train_data[:,dim]

X_test = test_data[:,0:dim]
y_test = test_data[:,dim]

X_train = X_train.reshape(X_train.shape[0] , 1 , 28 , 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0] , 1 , 28 , 28).astype('float32')

print("Training Data")
for i  in range(X_train.shape[0]):
    print(" ====> " , int(y_train[i]))
    print(i,X_train[i].shape,int(y_train[i]))
    display(X_train[i])

print("Testing Data")
for i  in range(X_test.shape[0]):
    print(" ====> " , int(y_test[i]))
    print(i,X_test[i].shape,int(y_test[i]))
    display(X_test[i])


y_train[0] = y_test[0] = 9
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = 10

print(X_train.shape , y_train.shape)
print(X_test.shape , y_test.shape)
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(180, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
    
    
model_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_digit.h5")
print("Saved model to disk")
    
    
    
    