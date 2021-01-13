import time
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
#            if((row < 4) or (row>24) or (col < 4) or (col > 24)):
#                img[0][row][col] = 0
            if ((img[0][row][col] >= threshold)):
                img[0][row][col] = 0
            else:
                img[0][row][col] = 1

def display(img, threshold=0.5):    
    image = cv2.imread('ZZZZZ_TestImg.png')
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
    

def num2Char(n):
    n = int(n)
    if(n>25):
        if(n == 26):
            return 'Space'
        elif(n == 27):
            return 'Space'
        elif(n == 28):
            return 'Space'
    else:
        return chr(ord('A') + n)

json_file = open('Models/model_alpha_coopes_bord_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelAlpha = model_from_json(loaded_model_json)
# load weights into new model
modelAlpha.load_weights("Models/model_alpha_coopes_bord_new.h5")
modelAlpha.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Alphabet Model loaded Successfully!!!')

num_alpha = 26

def alphaPredict(img):
    now = -1
    mx = -1
    arr = []
    arr.append(img)
    prediction = modelAlpha.predict(arr)
    pred = prediction[0]
    for i in range(num_alpha):
        if(pred[i] > mx):
            mx = pred[i]
            now = i
    ch = num2Char(now)
    return ch
    
#arr = [1,2,3,4]
#arr = np.array(arr)
#print(type(arr) , arr.shape)
#arr = arr.reshape(1,2,2).astype('float32')
#print(arr , arr.shape)


json_file = open('Models/model_digit_bord_merged.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelDig = model_from_json(loaded_model_json)
# load weights into new model
modelDig.load_weights("Models/model_digit_bord_merged.h5")
modelDig.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Digit Model loaded Successfully!!!')

num_dig = 10
def digitPredict(img):
    now = -1
    mx = -1
    arr = []
    arr.append(img)
    prediction = modelDig.predict(arr)
    pred = prediction[0]
    for i in range(num_dig):
        if(pred[i] > mx):
            mx = pred[i]
            now = i
    return chr(ord('0') + now)

json_file = open('Models/model_digit_012_bord.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelDig012 = model_from_json(loaded_model_json)
# load weights into new model
modelDig012.load_weights("Models/model_digit_012_bord.h5")
modelDig012.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Digit Model 012 loaded Successfully!!!')

def digitPredict012(img):
    now = -1
    mx = -1
    arr = []
    arr.append(img)
    prediction = modelDig012.predict(arr)
    pred = prediction[0]
    for i in range(3):
        if(pred[i] > mx):
            mx = pred[i]
            now = i
    return chr(ord('0') + now)