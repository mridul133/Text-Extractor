import F_FinalPreprocess

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob


def showImage(img, tag):
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.show()
    
    print(tag)
    
    return

def MakeSharp(img):
    
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(1,1))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l2 = clahe.apply(l)
    
    lab = cv2.merge((l2,a,b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img2

def Sharpen(img):
      
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    
    high=120
    
    i=0
    
    cnt=0
    
    tag = 'alpha'
    
    while i<height:
    
        j=0
        
        while j<width:
        
            if(img[i, j][0]>=0 and img[i, j][0]<high and img[i, j][1]>=0 and img[i, j][1]<high and img[i, j][2]>=0 and img[i, j][2]<high):
                img[i, j]=[0, 0, 0]
            else:
                img[i, j]=[255, 255, 255]
                
            if(i>6 and j>6 and i<height-5 and j<width-5 and img[i, j][0]==0):
                cnt+=1
                
            j = j+1
        i = i+1
    
    
    if(cnt<50):
        tag='space'
    
    else:
        if(cnt<100):
            tag='dot'
    
    return img, tag

def DeleteBorder(img):
    
    img, tag = Sharpen(img)
    
#    print(tag)
#    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0,0])
    upper_black = np.array([10,10,10])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    return mask, tag


def ActualPreProcess(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    img = img[4:height-3, 5:width-4] 
    
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    img, tag = DeleteBorder(img)
    
    img = 255-img
    img = F_FinalPreprocess.func5(img)

#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    img = 255-img
    
#    height = np.size(img, 0)
#    width = np.size(img, 1) 
#    
#    print(height, width)
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    img = cv2.resize(img, (28, 28))
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    return img, tag


def func3(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    hh=1
    ww=14
    
    d1 = height/hh
    d2 = width/ww
    
    splittedImg = []
    
    cnt=1
    
    i=0
    
    while i<height:
        j=0
        while j<width:
            
            i1 = (int)(i)
            j1 = (int)(j)
            i2 = (int)(i+d1)
            j2 = (int)(j+d2)
            
            if j2>width or i2>height: break
        
            img2 = img[i1:i2, j1:j2]
                        
#            plt.imshow(img2,cmap='gray',interpolation='bicubic')
#            plt.show()

            img2, tag = ActualPreProcess(img2)
            
#            plt.imshow(img2,cmap='gray',interpolation='bicubic')
#            plt.show()
    
            splittedImg.append((img2, tag))
            cnt+=1
            
            
            i = i1
            j = j1
            
            j=j+d2
            
        i=i+d1
    
    return splittedImg;


#
#image = cv2.imread("form3/1.jpg")
#new_image = func3(image, "name")