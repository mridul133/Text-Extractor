import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import glob

def preProcess1(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = np.size(img, 0)
    width = np.size(img, 1) 
     
    img = cv2.resize(img, (600, 800))
#     
    return img

def preProcess2(img):
    img = cv2.resize(img, (1200, 1600))
    #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    return img

def MakeSharp(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    
    return img


def getCount(image, x, y):
    cnt=0
    
    tot=0
    
    for i in range(x-15, x+15):
        for j in range(y-15, y+15):
            if i>=0 and j>=0 and i<800 and j<600:
                cnt+=1
                tot+=image[i][j]
#                tot+=(image[i][j][0]+image[i][j][1]+image[i][j][2])
#                if image[i][j][0]<130 and image[i][j][1]<130 and image[i][j][2]<130:
#                    cnt+=1
    
    return int(tot/cnt)

def getCorner(img, title):
    
    x1=0
    y1=0
    x2=0
    y2=0
    
    H = 800
    W = 600
    
    hor = 200
    vert = 200
    
    
    
    if title=="one":
        x1=1
        x2=71
        y1=1
        y2=141
        
    if title=="two":
        x1=1
        x2=vert+1
        y1=W-hor+1
        y2=W+1
        
    if title=="three":
        x1=H-vert+1
        x2=H+1
        y1=1
        y2=hor+1
        
    if title=="four":
        x1=H-vert+1
        x2=H+1
        y1=W-hor+1
        y2=W+1
    
    mat = np.zeros((H+20, H+20))
    
    for i in range(x1, x2):
        for j in range(y1, y2):
            mat[i][j]=img[i-1][j-1]+mat[i-1][j]+mat[i][j-1]-mat[i-1][j-1]
                
            
    mn = 100000000000000    
    arr= [] 
    for i in range(x1, x2):
        for j in range(y1, y2):
            
            LTx = max(i-7, x1)
            LTy = max(j-7, y1)
            RBx = min(i+7, x2-1)
            RBy = min(j+7, y2-1)
            
            cnt = mat[RBx][RBy]-mat[RBx][LTy-1]-mat[LTx-1][RBy]+mat[LTx-1][LTy-1]
            
            tot = (RBx-LTx+1)*(RBy-LTy+1)
            cnt=int(cnt/tot)
            
            if(title=="four"):
                arr.append((cnt, j, i))
            
            if cnt<mn:
                arr.append((cnt, j, i))
                mn=cnt
                
    arr.sort()
    
    return arr

def getFourCorners(img):
    
    scale=2;
    
    xs = []

    arr = getCorner(img, "one")
    
    for i in range(0, min(20, len(arr))):
        xs.append((scale*arr[i][1], scale*arr[i][2]))
        
    arr = getCorner(img, "two")
    
    for i in range(0, min(5, len(arr))):
        xs.append((scale*arr[i][1], scale*arr[i][2]))
    
    arr = getCorner(img, "three")
    
    for i in range(0, min(5, len(arr))):
        xs.append((scale*arr[i][1], scale*arr[i][2]))
    
    arr = getCorner(img, "four")
    
    for i in range(0, min(5, len(arr))):
        xs.append((scale*arr[i][1], scale*arr[i][2]))
    
    
#    cv2.namedWindow('croped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('croped', 600,700)
#    cv2.imshow("croped", img)
#    cv2.waitKey(0)
    
    tot=len(xs)
        
    xs.sort()
    
    points = np.zeros(shape=(tot,2))
    
    i=0
    
    while i<tot:
        points[i]=xs[i]
        i+=1
        

    return points
    
    

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


def show_detected(img, pnt):
    
#    cv2.circle(img,(40,70), 15, (0,0,255), 2)

    for i in range(0, len(pnt)):
        cv2.circle(img, (int(pnt[i][0]),int(pnt[i][1])), 15, (0,0,255), 4)
    
    cv2.namedWindow('out',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out', 600,700)
    cv2.imshow("out", img)
#    cv2.waitKey(0)

#    plt.imshow(im_with_keypoints,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    return;





def func7(image):
    
    image2 = preProcess1(image)
    points = getFourCorners(image2)
    
    image = cv2.resize(image, (1200, 1600))
    
    new_image = getNewImage(image, points)
    new_image = preProcess2(new_image)
    
#    show_detected(image, points)
#       
#    cv2.namedWindow('croped',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('croped', 600,700)
#    cv2.imshow("croped", new_image)
#    cv2.waitKey(0)
###    
#    plt.imshow(new_image,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    return new_image
    

#image = cv2.imread("form13/Alpha/Jhamela/003.jpg")
#new_image = func7(image)

#for img in glob.glob("2_CamScanned(Color)/Forms/Dig/*.jpg"): 
#    print(img)
#    image = cv2.imread(img)
#    new_image = func7(image)
