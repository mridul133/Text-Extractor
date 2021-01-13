import B_AlignAndCrop
import C_GetPortion
import D_Split
import E_MakeDataset
import F_FinalPreprocess
#import Api_Merged

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def display(img, img2):
    cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 600,700)
    cv2.imshow("img1", img)
    
    cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 600,700)
    cv2.imshow("img2", img2)
    
    cv2.waitKey(0)
    
    return

def showImage(img):
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.show()
    
    return


#image = cv2.imread("1_MobileCaptured/Forms/All/001.jpg")

for img in glob.glob("2_CamScanned(Color)/Forms/Alpha/005.jpg"):
   
    image = cv2.imread(img)

    new_image = B_AlignAndCrop.func1(image)
    
    AllCharacters = C_GetPortion.func2(new_image)
    
#    display(image, new_image)
    
    Dataset = []
    
    for i in range(0, len(AllCharacters)):
#        showImage(AllCharacters[i][0])
#        print(AllCharacters[i][1])
        Dataset.append((E_MakeDataset.getArr(AllCharacters[i][0]), AllCharacters[i][1]))
        
    


    
    
    