import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import F_FinalPreprocess


img = cv2.imread('TrainingImages/All/001.jpg')

img  = cv2.resize(img, (1200, 1600))


laplacian = cv2.Laplacian(img,cv2.CV_8UC1) # Laplacian OR
edges = cv2.Canny(img,100,10,apertureSize = 3) # canny Edge OR

cv2.namedWindow("a",cv2.WINDOW_NORMAL)
cv2.resizeWindow("a", 600,700)
cv2.imshow("a", edges)

# Hough's Probabilistic Line Transform 
minLineLength = 900
maxLineGap = 100
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


#
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
#cv2.namedWindow("a",cv2.WINDOW_NORMAL)
#cv2.resizeWindow("a", 600,700)
#cv2.imshow("a", edges)
#
#lines = cv2.HoughLines(edges,1,np.pi/180,200)
#
#for i in range(0, len(lines)):
#    for rho,theta in lines[i]:
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a*rho
#        y0 = b*rho
#        x1 = int(x0 + 1000*(-b))
#        y1 = int(y0 + 1000*(a))
#        x2 = int(x0 - 1000*(-b))
#        y2 = int(y0 - 1000*(a))
#    
#        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


cv2.namedWindow("b",cv2.WINDOW_NORMAL)
cv2.resizeWindow("b", 600,700)
cv2.imshow("b", img)
cv2.waitKey(0)

plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.show()