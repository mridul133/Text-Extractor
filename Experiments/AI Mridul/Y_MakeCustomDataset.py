import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import F_FinalPreprocess
import D_Split


def MakeSharp(img):
    
    img = cv2.adaptiveThreshold(img, 55, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    
    return img

def Sharpen(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    i=0
    
    while i<height:
    
        j=0
        
        while j<width:
            px = img[i, j]
            
            if(px>170):
                img[i, j]=255
            else:
                img[i, j]=0
            
            j = j+1
        
        i = i+1
        
    
    return img


def getArr(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1) 
    i=0
    
    arr = []
    
    
    while i<height:
    
        j=0
        
        while j<width:
            px = img[i, j]
#            print(px)
            if px<120:
                px=0
                img[i, j]=0
            else:
                px=255
                img[i, j]=255
#            
            px = px/255
            
            
            
            arr.append(px)
            
            j = j+1
        
        i = i+1
    
#    print(string)


#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()

    return arr

#Main

def PreProcessMajhe(img):
    
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    img = D_Split.DeleteBorder(img)
    
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    
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
#    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    return img

def PreProcessWithoutBorder(img):
    
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    img = D_Split.DeleteBorder(img)
    
#    img = 255-img
    
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    img = cv2.resize(img, (28, 28))
#    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
#    
    return img

def PreProcessNormal(img):
    
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = 255-img
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    img = cv2.resize(img, (28, 28))
    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
#    
    return img


def getFoldName(n):
    
    ret="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n==26:
        return "Hyphen"
    if n==27:
        return "Dot"
    if n==28:
        return "Space"
    return ret[n]


f1 = open('rahul_train.csv','w')
f2 = open('rahul_test.csv','w')

for i in range(0, 29):
    
    cnt=0
    
    foldName = getFoldName(i)
    print(foldName)
    for img in glob.glob("1_MobileCaptured/Alphabets/"+str(foldName)+"/*.jpg"):
        
        cnt+=1
        
        cv_img = cv2.imread(img)
        
        height = np.size(cv_img, 0)
        width = np.size(cv_img, 1)
        
        cv_img = cv_img[4:height-4, 4:width-4] 
        
#        plt.imshow(cv_img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        cv_img = PreProcessNormal(cv_img)
#        cv_img = PreProcessWithoutBorder(cv_img)
#        cv_img = PreProcessMajhe(cv_img)
        
#        plt.imshow(cv_img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        

        arr = getArr(cv_img)

#        print(cnt)

        if cnt>160:
           
            for j in range(0, len(arr)):
            
                f1.write(str(arr[j])+', ')
                j+=1
            
            name = i
            
            f1.write(str(name)+'\n')
        
        
        else:
    
            for j in range(0, len(arr)):
            
                f2.write(str(arr[j])+', ')
                j+=1
            
            name = i
        
        
            f2.write(str(name)+'\n')
#            

f1.close()
f2.close()