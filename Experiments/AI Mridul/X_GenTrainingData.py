import B_AlignAndCrop
import C_GetPortion
import D_Split
import E_MakeDataset
import W_MakeTrainigFiles
import G_GollaDetect
import Y_MakeCustomDataset

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob

def getFoldName(n):
    
    ret="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    return ret[n]

def clearFolder():
    
    files = glob.glob('OurData/Dot/*')
    
    for f in files:
        os.remove(f)
        
    files = glob.glob('OurData/Space/*')
    for f in files:
        os.remove(f)
        
    files = glob.glob('OurData/Hyphen/*')
    for f in files:
        os.remove(f)
    
    for i in range(0, 26):
        
        files = glob.glob('OurData/'+str(getFoldName(i)+"/*"))
        
        for f in files:
            os.remove(f)
            
    return;

def clearFolderDig():
    
    for i in range(0, 10):
        
        files = glob.glob('DigitData/'+str(i)+"/*")
        
        for f in files:
            os.remove(f)
            
    return;

def getPart(img, title):
    
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    scale=1
    
    LTx=0
    LTy=0
    RBx=0
    RBy=0
    
    
    if title=="one":
        LTx=scale*30
        LTy=scale*325
        RBx=scale*1172
        RBy=scale*378
    
    if title=="two":
        LTx=scale*30
        LTy=scale*380
        RBx=scale*1172
        RBy=scale*438
    
    if title=="three":
        LTx=scale*30
        LTy=scale*440
        RBx=scale*1172
        RBy=scale*500
    
    if title=="four":
        LTx=scale*30
        LTy=scale*499
        RBx=scale*1172
        RBy=scale*555
        
    if title=="five":
        LTx=scale*32
        LTy=scale*558
        RBx=scale*1172
        RBy=scale*615  
    
    if title=="six":
        LTx=scale*32
        LTy=scale*615
        RBx=scale*1172
        RBy=scale*672
        
    if title=="seven":
        LTx=scale*33
        LTy=scale*670
        RBx=scale*1172
        RBy=scale*729
        
    if title=="eight":
        LTx=scale*34
        LTy=scale*729
        RBx=scale*1172
        RBy=scale*784
        
    if title=="nine":
        LTx=scale*35
        LTy=scale*786
        RBx=scale*1170
        RBy=scale*843
        
    if title=="ten":
        LTx=scale*35
        LTy=scale*842
        RBx=scale*1170
        RBy=scale*902
        
    if title=="eleven":
        LTx=scale*35
        LTy=scale*902
        RBx=scale*1170
        RBy=scale*958
        
    if title=="twelve":
        LTx=scale*37
        LTy=scale*1102
        RBx=scale*1170
        RBy=scale*1157
        
    if title=="thirteen":
        LTx=scale*37
        LTy=scale*1190
        RBx=scale*1173
        RBy=scale*1246
        
    if title=="fourteen":
        LTx=scale*37
        LTy=scale*1278
        RBx=scale*1173
        RBy=scale*1335
    
    if title=="fifteen":
        LTx=scale*37
        LTy=scale*1367
        RBx=scale*1173
        RBy=scale*1423
    
    if title=="sixteen":
        LTx=scale*37
        LTy=scale*1457
        RBx=scale*1173
        RBy=scale*1513
        
        
#    oldImg = img
    
#    cv2.circle(oldImg,(LTx,LTy), 10, (0,0,255), -1)
#    cv2.circle(oldImg,(RBx,RBy), 10, (0,0,255), -1)
    
#    if title=="sixteen":
        
    img = img[LTy:RBy, LTx:RBx]

#    cv2.namedWindow('old',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('old', 600,700)
#    cv2.imshow("old", oldImg)
#    
#    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
#    cv2.resizeWindow(title, 600,70)
#    cv2.imshow(title, img)
#    cv2.waitKey(0)
#    
#    plt.imshow(img,cmap='gray',interpolation='bicubic')
#    plt.show()
#    
    return img




#clearFolder()
#clearFolderDig()

#
#for img in glob.glob("2_CamScanned(Color)/Forms/Dig/*.jpg"):
#    
#    image = cv2.imread(img)
#        
#    image = G_GollaDetect.func7(image)
#    
#    height = np.size(image, 0)
#    width = np.size(image, 1)
#    
#    part = [None]*17
#    
#    part[1] = getPart(image, "one")
#    part[2] = getPart(image, "two")
#    part[3] = getPart(image, "three")
#    part[4] = getPart(image, "four")
#    part[5] = getPart(image, "five")
#    part[6] = getPart(image, "six")
#    part[7] = getPart(image, "seven")
#    part[8] = getPart(image, "eight")
#    part[9] = getPart(image, "nine")
#    part[10] = getPart(image, "ten")
#    part[11] = getPart(image, "eleven")
#    part[12] = getPart(image, "twelve")
#    part[13] = getPart(image, "thirteen")
#    part[14] = getPart(image, "fourteen")
#    part[15] = getPart(image, "fifteen")
#    part[16] = getPart(image, "sixteen")
#    
#    
#    SplittedImages = []
#    
#    for k in range(1, 17):
#        temp = D_Split.func3(part[k], "test")
#        for j in range(0, len(temp)):
#            i=(k-1)*14+j
#         
#            if (i>8 and i<14) or (i>22 and i<28):
#                continue
#            if (i>36 and i<42) or (i==51) or (i>55 and i<60) or (i>64 and i<70) or (i==79) :
#                continue
#            if (i>83 and i<89) or (i==98) or (i==99) or (i==107) or (i==108) or (i==110) or (i==112) or (i==113) or (i==116) or (i==117):
#                continue
#            if (i>120 and i<123) or (i>125 and i<129) or (i>139 and i<143):
#                continue
#            
#            SplittedImages.append(temp[j])
#            
#            
#            
##            img = Y_MakeCustomDataset.PreProcessNormal(temp[j])
##            img = Y_MakeCustomDataset.PreProcessWithoutBorder(temp[j])
##            img = Y_MakeCustomDataset.PreProcessMajhe(temp[j])
#    
# 
##    W_MakeTrainigFiles.OKupto_62(SplittedImages)
##    W_MakeTrainigFiles.OKupto_64(SplittedImages)
##    W_MakeTrainigFiles.All(SplittedImages)
##    W_MakeTrainigFiles.AllExcept_2(SplittedImages)
##    W_MakeTrainigFiles.Digits(SplittedImages)
#
#
#
##plt.imshow(new_image,cmap='gray',interpolation='bicubic')
##plt.show()