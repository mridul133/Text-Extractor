import cv2
from sys import argv
from scipy import stats
import numpy as np
import sys
import os
import math
import argparse
import matplotlib.pyplot as plt
import glob


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
            
            
    return Image_Thinned



# border removal by inpainting
def border_removal(box_bw,top,bottom,right,left):    
    box_bw[0:top,:]=255   # first "top"  number of rows
    box_bw[-bottom:,]=255 # last "bottom" number of rows
    box_bw[:,0:left]=255  # first "left" number of columns
    box_bw[:,-right:]=255 # last "right" number of columns
    # last two rows a[-2:,]
    return box_bw

def remove_line(box_bw,line_thickness):
    edges = cv2.Canny(box_bw, 80, 120)

    # dilate: it will fill holes between line segments 
    (r,c)=np.shape(box_bw)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
    edges=cv2.dilate(edges,element)
    min=np.minimum(r,c)
    lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, min*0.75, 1);
        
    r_low_lim=r*0.1
    r_high_lim=r-r_low_lim

    c_low_lim=c*0.1
    c_high_lim=c-c_low_lim

    if lines!=None:
        for line in lines[0]:
            pt1 = (line[0],line[1])
            pt2 = (line[2],line[3])                 
            theta_radian2 = np.arctan2(line[2]-line[0],line[3]-line[1]) #calculating the slope and the result returned in radian!
            theta_deg2 = (180/math.pi)*theta_radian2 # converting radian into degrees!
            if (theta_deg2>85 and theta_deg2<95): # horizontal line                
                # if starting of line is below or above 30% of box, remove it
                if (line[1]<=r_low_lim or line[1]>=r_high_lim) and (line[3]<=r_low_lim or line[3]>=r_high_lim):
                    cv2.line(box_bw, pt1, pt2, 255, line_thickness)        
            if(theta_deg2>175 and theta_deg2<185):# vertical line
                if (line[0]<=c_low_lim or line[0]>=c_high_lim) and (line[2]<=c_low_lim or line[2]>=c_high_lim):
                    cv2.line(box_bw, pt1, pt2, 255, line_thickness)                    
    return box_bw

# Function that will do the main job
def func5(original_image):
       #reading captured (scanned image)

    #image thresholding to make binary image of the box
    (thresh, box_bw) = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    noise_removal = 'YES'
    remove_border_size = 1
    #border removal
    top=remove_border_size
    bottom=remove_border_size
    right=remove_border_size
    left=remove_border_size
#    box_bw_border_free=border_removal(box_bw,top,bottom,right,left)
    box_bw_border_free=box_bw

    # MEDIAN FILTERING : (e.g. Salt and pepper type noise will be removed) 
    if noise_removal.startswith('Y') or noise_removal.startswith('y'):
        box_bw_border_free=cv2.medianBlur(box_bw_border_free,3)
        
    H1 = np.size(box_bw_border_free, 0)
    W1 = np.size(box_bw_border_free, 1)
    
#    print("pre ", H, W)

    # auto-crop out whitespace
    (thresh, In_bw) = cv2.threshold(box_bw_border_free,128, 255, cv2.THRESH_BINARY) # thresholding
    inverted_In_bw=np.invert(In_bw) # inverted so that black becomes white and white becomes black since we will check for nonzero values
    (i,j)=np.nonzero(inverted_In_bw) # finding indexes of nonzero values
    if np.size(i)!=0: # in case the box contains no BLACK pixel(i.e. the box is empty such as checkbox)
        Out_cropped = box_bw_border_free[np.min(i):np.max(i),np.min(j):np.max(j)] # row column operation to extract the non
    else: # no need to do cropping since its an empty box
         Out_cropped = box_bw_border_free
         
         
         
    height = 50
    width = 50
    
    H = np.size(Out_cropped, 0)
    W = np.size(Out_cropped, 1)
    
    if H==0 or W==0:
        Out_cropped = box_bw_border_free
    else:
        if (W/H)<0.4 or (H/W)<0.4 or (H/H1)<0.25 or (W/W1)<0.25:
            border_width = 0.1
            box_bw_thinned_bordered = cv2.copyMakeBorder(Out_cropped, int(height*border_width), int(height*border_width), int(width*border_width), int(width*border_width), cv2.BORDER_CONSTANT, value=255)  
            struc_element = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
            Out_cropped=cv2.erode(box_bw_thinned_bordered,struc_element)    
    
#    print(height, width)
    
    Ithin_resized=cv2.resize(Out_cropped,(width,height),None,0,0,cv2.INTER_LANCZOS4) 

    #-------PRE-THIN THRESHOLDING----------
    #image thresholding to make binary image of the thinned image
    (thresh, Ithin_resized_thresh) = cv2.threshold(Ithin_resized,200, 255, cv2.THRESH_BINARY)
    # for debugging
    #cv2.imwrite(fname_box+'.Ithin_resized_thresh.png', Ithin_resized_thresh)

    #-------ZHANG-SUEN-THINNING------------
#    Ithin_resized_thresh = 255-Ithin_resized_thresh
    box_bw_thinned=(Ithin_resized_thresh)
#    box_bw_thinned = 255-box_bw_thinned
    
#    plt.imshow(box_bw_thinned,cmap='gray',interpolation='bicubic')
#    plt.show()
    # For debugging
    #cv2.imwrite(fname_box+'.box-bw-thinned-no-erosion.png', box_bw_thinned)

    #------ADD SPACING BORDER---------------------------
    # To create space for line expansion via erosion
    border_width = 0.3 # as a % of the total image dimensions
    box_bw_thinned_bordered = cv2.copyMakeBorder(box_bw_thinned, int(height*border_width), int(height*border_width), int(width*border_width), int(width*border_width), cv2.BORDER_CONSTANT, value=255)  

    #-----------EROSION--------------------------
    #apply some erosion to join the gaps
    struc_element = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    Output=cv2.erode(box_bw_thinned_bordered,struc_element)
    
    
#    plt.imshow(Output,cmap='gray',interpolation='bicubic')
#    plt.show()
    
    return Output


#for img in glob.glob("Images/2/*.png"):
#       
#    cv_img = cv2.imread(img, 0)
#    
#    plt.imshow(cv_img,cmap='gray',interpolation='bicubic')
#    plt.show()
#
#    
#    func5(cv_img)
#               
