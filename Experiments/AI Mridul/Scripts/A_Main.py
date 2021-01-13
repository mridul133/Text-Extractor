import B_AlignAndCrop
import C_GetPortion
import D_Split
import E_MakeDataset
import F_FinalPreprocess
import G_GetPrediction

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os 

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

def Normalize(s):
    
    l = len(s)-1
    i=0
    
    while s[i]==' ':
        i+=1
    
    while s[l]==' ':
        l-=1
    

    s = s[i:l+1]

    return s    
    


#image = cv2.imread("2_CamScanned(Color)/Forms/Alpha/005.jpg")

for img in glob.glob("../Inputs/*.jpg"):
   
    image = cv2.imread(img)
    
    new_image = B_AlignAndCrop.func1(image)

#    display(image, new_image)
    
    AllCharacters = C_GetPortion.func2(new_image)
   
    Dataset = []
    
    for i in range(0, len(AllCharacters)):
#        showImage(AllCharacters[i][0])
        Dataset.append((E_MakeDataset.getArr(AllCharacters[i][0]), AllCharacters[i][1]))
        
#
#
#
#
#    dataset = np.array(Dataset)
#    
#    
#    FinalString = ''
#    
#
#    for i in range(dataset.shape[0]):
##        if(dataset[i][1] != 'alpha'):
##            continue
##        j = i%10
##        if(j > 2):
##            continue
#
#        x = dataset[i][0]
#        x = np.array(x)
##        print(x , x.shape)
#        x = x.reshape(1,1,28,28).astype('float32')
#    #    print(x , x.shape)
#        G_GetPrediction.modify(x[0])
##        G_GetPrediction.display(x[0])
#        
#        ch = ' '
#        
#        if(dataset[i][1]=='dot'):
#            ch = '.'
#        if(dataset[i][1]=='alpha'):
#            ch = G_GetPrediction.alphaPredict(x)
#        if(dataset[i][1]=='dig10'):
#            ch = G_GetPrediction.digitPredict(x)
#        if(dataset[i][1]=='dig3'):
#            ch = G_GetPrediction.digitPredict012(x)
##        print(' ===> ' , ch)
#        
#        FinalString+=ch
#       
#
##    print(FinalString, len(FinalString))
#    
#    
#    
#    Name = Normalize(FinalString[0:36])
#    ParName = Normalize(FinalString[40:54])
#    DOB = FinalString[54:56]+'/'+FinalString[56:58]+'/'+FinalString[36:40]
#    Phone = '01'+FinalString[58:67]
#    EIIN = Normalize(FinalString[67:74])
#    Status = FinalString[74]
#    Sex = FinalString[75]
#    Level = Normalize(FinalString[76:78])
#    Section = Normalize(FinalString[78:81])
#    Roll = Normalize(FinalString[81:84])
#    SubCode = Normalize(FinalString[84:95])
#    ParPhone = '01' + FinalString[95:104]
#    Seventy = FinalString[104:174]
#    ProPic = C_GetPortion.getPart(new_image, 'proPic')
#    Sign = C_GetPortion.getPart(new_image, 'sign')
#    Date = C_GetPortion.getPart(new_image, 'date')
#    
#    
#    
#    
##    print(Name, ParName, DOB, Phone, EIIN, Status, Sex, Level, Section, Roll, SubCode, ParPhone, Seventy)
##    showImage(ProPic)
##    showImage(Sign)
##    showImage(Date)
##    
#    
#    
#    path = '../Outputs/'+img[10 : int(len(img)-4)]
#    
#    if not os.path.exists(path):
#        os.makedirs(path)
#    
#    f = open(path+'/Info.txt', 'w')
#    
#    f.write('Name : ' + Name + '\n')
#    f.write('Mother\'s/Father\'s Name : ' + ParName + '\n')
#    f.write('Date Of Birth : ' + DOB + '\n')
#    f.write('Personal Cell No : ' + Phone + '\n')
#    f.write('EIIN : ' + EIIN + '\n')
#    f.write('Status : ' + Status + '\n')
#    f.write('Gender : ' + Sex + '\n')
#    f.write('Level : ' + Level + '\n')
#    f.write('Section : ' + Section + '\n')
#    f.write('Roll No : ' + Roll + '\n')
#    f.write('Sub/Group Code : ' + SubCode + '\n')
#    f.write('Parent\'s Cell No : ' + ParPhone + '\n')
#    f.write('Opinions : ' + Seventy + '\n')
#    
#    f.close()
#    
#    cv2.imwrite(path+'/date.png', Date)
#    cv2.imwrite(path+'/signature.png', Sign)
#    cv2.imwrite(path+'/photo.png', ProPic)
#    
#    
#    print('Done !!')
#    