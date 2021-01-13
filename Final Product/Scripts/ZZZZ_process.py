import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def preProcess1(img):
     height = np.size(img, 0)
     width = np.size(img, 1) 
     
     rat = height/width
     
     img = cv2.resize(img, (1200, int(1200*rat)))
     
     return img

def preProcess2(img):
    img = cv2.resize(img, (1200, 1600))
    #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    return img

def MakeSharp(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    
    return img



def show_detected(img, params):
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.namedWindow('out',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out', 600,700)
    cv2.imshow("out", im_with_keypoints)
    
    return;

def getFourCorners(img, params):
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    xs = []
    
    tot = len(keypoints)
    
#    print(tot)
    
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        
        xs.append((x, y))
        
    xs.sort()
    
#    print(xs)
    
    points = np.zeros(shape=(tot,2))
    
    i=0
    
    while i<tot:
        points[i]=xs[i]
        i+=1
        
    hull = ConvexHull(points)
    vtc = hull.vertices
    
    
    
#    print("length")
#    print(len(vtc))
    
    i=0
    ln = len(vtc)
    
    hull_points = np.zeros(shape=(ln,2))
    
    while i<ln:
        hull_points[i]=points[vtc[i]]
        i+=1;

    
#    hull_points = hull_points.reshape((-1,1,2))
#    img = cv2.polylines(img, np.int32([hull_points]), True, (0,255,255),3)
#    
#    cv2.namedWindow('out2',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('out2', 600,700)
#    cv2.imshow("out2", img)
#    cv2.waitKey(0)
    
    return hull_points
    
    

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	return rect


def getNewImage(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	return warped
        

def getName(img):
    cv2.circle(new_image,(60,295), 5, (0,0,255), -1)
    cv2.circle(new_image,(745,532), 5, (0,0,255), -1)
    
#    cv2.namedWindow('old',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('old', 600,700)
#    cv2.imshow("old", img)
    
    img = img[295:532, 60:745]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,300)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getParentsName(img):
    cv2.circle(new_image,(63,526), 5, (0,0,255), -1)
    cv2.circle(new_image,(745,646), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[526:646, 63:745]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,150)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getPhoneNo(img):
    cv2.circle(new_image,(293,644), 5, (0,0,255), -1)
    cv2.circle(new_image,(1137,705), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[644:705, 293:1137]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getEIIN(img):
    cv2.circle(new_image,(218,705), 5, (0,0,255), -1)
    cv2.circle(new_image,(745,760), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[705:760, 218:745]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getLevel(img):
    cv2.circle(new_image,(218,757), 5, (0,0,255), -1)
    cv2.circle(new_image,(368,815), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[757:815, 218:368]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 120,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getSection(img):
    cv2.circle(new_image,(522,759), 5, (0,0,255), -1)
    cv2.circle(new_image,(745,815), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[759:815, 522:745]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 180,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getRoll(img):
    cv2.circle(new_image,(903,759), 5, (0,0,255), -1)
    cv2.circle(new_image,(1136,815), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[759:815, 903:1136]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 180,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getGroupCode(img):
    cv2.circle(new_image,(293,817), 5, (0,0,255), -1)
    cv2.circle(new_image,(1137,875), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[817:875, 293:1137]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
    return img

def getParentsPhoneNo(img):
    cv2.circle(new_image,(293,872), 5, (0,0,255), -1)
    cv2.circle(new_image,(1137,930), 5, (0,0,255), -1)
    
#    cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Warped', 600,700)
#    cv2.imshow("Warped", img)
    
    img = img[872:930, 293:1137]
    
#    cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('new', 600,50)
#    cv2.imshow("new", img)
#    cv2.waitKey(0)
    
    
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
        
            if(px>100):
                img[i, j]=0
                px=0
            else:
                img[i, j]=255
                px=255
                
            arr.append(px/255)
            
            j = j+1
        
        i = i+1
    
    return img, arr



def SplitName(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    d1 = height/4
    d2 = width/9
    
    splittedImg = []
    
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
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, (28, 28))
            plt.imshow(img2,cmap='gray',interpolation='bicubic')
            plt.show()
            img2, arr=getArr(img2)
            splittedImg.append(arr)
            #print(arr)
            plt.imshow(img2,cmap='gray',interpolation='bicubic')
            plt.show()
            
            i = i1
            j = j1
            
            j=j+d2
            
        i=i+d1
    
    #print(splittedImg)
    
    f = open('Api5Name.txt','w')
    
    for i in range(0, len(splittedImg)):
    
        for j in range(0, len(splittedImg[i])):
        
            f.write(str(splittedImg[i][j])+', ')
            j=j+1
        
        i=i+1
        name = str(i)
        
        f.write(str(name)+'\n')
    
    f.close()
    
    return;

def SplitPhoneNo(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    d1 = height
    d2 = width/11
    
    splittedImg = []
    
    cnt=0
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
            
            
            
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, (28, 28))
            img2, arr=getArr(img2)
            
            #cv2.imwrite('digitImages/' + str(cnt) + '.png', img2)
            cnt+=1
            splittedImg.append(arr)
            #print(arr)
            plt.imshow(img2,cmap='gray',interpolation='bicubic')
            plt.show()
            
            i = i1
            j = j1
            
            j=j+d2
            
        i=i+d1
    
    #print(splittedImg)
    
    f = open('Api8Dig.txt','w')
    
    for i in range(0, len(splittedImg)):
    
        for j in range(0, len(splittedImg[i])):
        
            f.write(str(splittedImg[i][j])+', ')
            j=j+1
        
        i=i+1
        name = str(i)
        
        f.write(str(name)+'\n')
    
    f.close()
    
    return;




def getParams():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 5
    params.filterByArea = True
    params.minArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    return params


#Main

image = cv2.imread("form4/7.jpg")


#cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Warped', 600,700)
#cv2.imshow("Warped", image)
#
#cv2.namedWindow('new',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('new', 600,700)
#cv2.imshow("new", image2)


image = preProcess1(image)
#image = MakeSharp(image)

params = getParams()
points = getFourCorners(image, params)
new_image = getNewImage(image, points)
new_image = preProcess2(new_image)

#show_detected(image, params)

#img_name = getName(new_image)
#plt.imshow(img_name,cmap='gray',interpolation='bicubic')
#plt.show()
#
##img_parName = getParentsName(new_image)
##plt.imshow(img_parName,cmap='gray',interpolation='bicubic')
##plt.show()
##
img_phoneNo = getPhoneNo(new_image)
plt.imshow(img_phoneNo,cmap='gray',interpolation='bicubic')
plt.show()
#
#
#img_EIIN = getEIIN(new_image)
#plt.imshow(img_EIIN,cmap='gray',interpolation='bicubic')
#plt.show()
#
#img_Level = getLevel(new_image)
#plt.imshow(img_Level,cmap='gray',interpolation='bicubic')
#plt.show()
#
#img_Section = getSection(new_image)
#plt.imshow(img_Section,cmap='gray',interpolation='bicubic')
#plt.show()
#
#img_Roll = getRoll(new_image)
#plt.imshow(img_Roll,cmap='gray',interpolation='bicubic')
#plt.show()
#
#img_GroupCode = getGroupCode(new_image)
#plt.imshow(img_GroupCode,cmap='gray',interpolation='bicubic')
#plt.show()
#
#img_ParentsPhoneNo = getParentsPhoneNo(new_image)
#plt.imshow(img_ParentsPhoneNo,cmap='gray',interpolation='bicubic')
#plt.show()
#
#SplitName(img_name)
SplitPhoneNo(img_phoneNo)

#cv2.namedWindow('Warped',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Warped', 600,700)
#cv2.imshow("Warped", new_image)

cv2.waitKey(0)