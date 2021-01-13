import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob
import F_FinalPreprocess
import os


WriteDir ="2_CamScanned(Color)/Digits/"


def getFoldName(n):
    
    ret="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    return ret[n]


def OKupto_62(imgArr):
        
    cnt=0
    
    k=1
    
    for i in range(0, len(imgArr)):
        
        img = imgArr[i]
        
        list = os.listdir(WriteDir+str(getFoldName(cnt)))
        tot = len(list)
        
        path = WriteDir+str(getFoldName(cnt))+"/"+str(tot+1)+".jpg"
        
        cv2.imwrite(path, img)
        
#        plt.imshow(img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        print(k, path)
        
        k+=1
        if(k==157):
            break
        cnt+=1
        if(cnt==26):
            cnt=0
            
    
    for i in range(156, 162):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Hyphen")
        tot = len(list)
        path = WriteDir+"Hyphen/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
        
    for i in range(162, 168):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Dot")
        tot = len(list)
        path = WriteDir+"Dot/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
        
    for i in range(168, 176):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Space")
        tot = len(list) 
        path = WriteDir+"Space/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
        
    return;


def OKupto_64(imgArr):
        
    cnt=0
    
    j=0
    
    k=1
    
    for i in range(0, len(imgArr)):
        
        if i==74 or i==75:
            continue
        
        img = imgArr[i]
        
        list = os.listdir(WriteDir+str(getFoldName(cnt)))
        tot = len(list)
        
        path = WriteDir++str(getFoldName(cnt))+"/"+str(tot+1)+".jpg"
        
        cv2.imwrite(path, img)
        
#        plt.imshow(img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        print(path)
        
        k+=1
        if(k==157):
            break
        cnt+=1
        if(cnt==26):
            cnt=0
            
            
    for i in range(158, 164):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Hyphen")
        tot = len(list)
        path = WriteDir+"Hyphen/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
        
    for i in range(164, 170):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Dot")
        tot = len(list)
        path = WriteDir+"Dot/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
        
    for i in range(170, 176):
        img = imgArr[i]
        list = os.listdir(WriteDir+"Space")
        tot = len(list) 
        path = WriteDir+"Space/"+str(tot+1)+".jpg"
        print(path)
        cv2.imwrite(path, img)
            
    return;


def All(imgArr):
        
    cnt=0
    
    for i in range(0, len(imgArr)):
       
        img = imgArr[i]
        
        list = os.listdir(WriteDir+str(getFoldName(cnt)))
        tot = len(list)
        
        path = WriteDir+str(getFoldName(cnt))+"/"+str(tot+1)+".jpg"
        
        cv2.imwrite(path, img)
        
#        plt.imshow(img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        print(path)
        
        cnt+=1
        if(cnt==26):
            cnt=0
            
    return;

def AllExcept_2(imgArr):
        
    cnt=0
    
    for i in range(0, len(imgArr)):
        
        if i==74 or i==75:
            continue
        
        img = imgArr[i]
        
        list = os.listdir(WriteDir+str(getFoldName(cnt)))
        tot = len(list)
        
        path = WriteDir+str(getFoldName(cnt))+"/"+str(tot+1)+".jpg"
        
        cv2.imwrite(path, img)
        
#        plt.imshow(img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        print(path)
       
        cnt+=1
        if(cnt==26):
            cnt=0
            
    return;


def Digits(imgArr):
        
    cnt=0
    
    for i in range(0, len(imgArr)):
       
        img = imgArr[i]
        
        list = os.listdir(WriteDir+str(cnt))
        tot = len(list)
        
        path = WriteDir+str(cnt)+"/"+str(tot+1)+".jpg"
        
        cv2.imwrite(path, img)
        
#        plt.imshow(img,cmap='gray',interpolation='bicubic')
#        plt.show()
        
        print(path)
        
        cnt+=1
        if(cnt==10):
            cnt=0
            
    return;