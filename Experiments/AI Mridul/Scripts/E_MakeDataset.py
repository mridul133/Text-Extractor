import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob
import F_FinalPreprocess

def getArr(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    i=0
    
    arr = []
    
    while i<height:
    
        j=0
        
        while j<width:
            px = img[i, j]
        
            if(px>100):
                img[i, j]=0
                px=0
            else:
                img[i, j]=255
                px=255
                
            arr.append(px/255)
            
            j = j+1
        
        i = i+1
        
#    print(len(arr), "ok")
    
    return arr

def Sharpen(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    i=0
    
    while i<height:
    
        j=0
        
        while j<width:
        
            if(img[i, j]>180):
                img[i, j]=0
            else:
                img[i, j]=255
            
            j = j+1
        
        i = i+1
    
    return img


def PreProcess(img):
    
#    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #img = Sharpen(img)
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    img = 255-img
    

    img = F_FinalPreprocess.func5(img)
    
    img = 255-img
    
#    img = cv2.resize(img, (28, 28))
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
#    
    return img

def func4(splittedImg, fileName):
    
    arrays = []
    
    for i in range(0, len(splittedImg)):
        splittedImg[i] = PreProcess(splittedImg[i])
        
        img1, arr1 = getArr(splittedImg[i])
        
        arrays.append(arr1)
        i+=1
    
    
#    print(arrays)
    
#    print(len(arrays))
    
#    f = open(fileName,'w')
#    
#    for i in range(0, len(arrays)):
#    
#        for j in range(0, len(arrays[i])):
#            
#            f.write(str(arrays[i][j])+', ')
#            j=j+1
#        
#        i=i+1
#        name = str(i)
#        
#        f.write(str(name)+'\n')
#    
#    f.close()
#    
    return



#img_arr = []
#
#for img in glob.glob("Images/imgDig/*.png"):
#    
#    image = cv2.imread(img)
#    img_arr.append(image)
#
#
#func4(img_arr, "filename.txt")
                
                
                
                