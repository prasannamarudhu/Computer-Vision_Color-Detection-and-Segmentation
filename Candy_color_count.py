# Python program for Detection of a specific color(blue here) using OpenCV with Python 

import cv2 
import numpy as np  
from skimage import morphology

OrigImg = cv2.imread("candy.jpg")


# Converts images from BGR to HSV
hsv = cv2.cvtColor(OrigImg,cv2.COLOR_BGR2HSV) 

# Here we are defining range of bluecolor in HSV 
lower_blue = np.array([90,50,50]) 
upper_blue = np.array([110,255,255]) 

# Here we are defining range of greencolor in HSV 
lower_green = np.array([30,5,5]) 
upper_green = np.array([85,255,255]) 

# Here we are defining range of redcolor in HSV 
lower_red = np.array([160,70,70]) 
upper_red = np.array([180,255,255]) 

# Here we are defining range of yellowcolor in HSV 
lower_yellow = np.array([19,90,90]) 
upper_yellow = np.array([30,255,255]) 

# Here we are defining range of blackcolor in HSV 
lower_black = np.array([0,0,0]) 
upper_black = np.array([25,200,150]) 

# Here we are defining range of orangecolor in HSV 
lower_orange = np.array([2,20,20]) 
upper_orange = np.array([17,255,255]) 

# This creates a mask of coloured objects found in the Image. 
mask_RED = cv2.inRange(hsv, lower_red, upper_red)
mask_BLUE = cv2.inRange(hsv, lower_blue, upper_blue)
mask_GREEN = cv2.inRange(hsv, lower_green, upper_green)
mask_YELLOW = cv2.inRange(hsv, lower_yellow, upper_yellow) 
mask_BLACK = cv2.inRange(hsv, lower_black, upper_black) 
mask_ORANGE = cv2.inRange(hsv, lower_orange, upper_orange)
 
# The bitwise and of the frame and mask is done so that only the blue coloured objects are highlighted and stored in res 

#Red:
#result
res_RED = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_RED) 
FiltMedian_RED=cv2.medianBlur(res_RED,15)  # Filter, Smooth, Blur
#Count and print
gray_RED = cv2.cvtColor(FiltMedian_RED,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
res_RED = cv2.morphologyEx(gray_RED,cv2.MORPH_OPEN,kernel)

ret , threshold_RED = cv2.threshold(res_RED,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursRED,_=cv2.findContours(threshold_RED,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Red: %d  "%len(contoursRED)
cv2.drawContours(FiltMedian_RED,contoursRED,-1,(0,255,255),2)
#cv2.namedWindow('Display1',cv2.WINDOW_NORMAL)

#Green:
#result
res_GREEN = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_GREEN) 
FiltMedian_GREEN=cv2.medianBlur(res_GREEN,15)  # Filter, Smooth, Blur
#Count and print
gray_GREEN = cv2.cvtColor(FiltMedian_GREEN,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
res_GREEN = cv2.morphologyEx(gray_GREEN,cv2.MORPH_OPEN,kernel)

ret , threshold_GREEN = cv2.threshold(res_GREEN,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursGREEN,_=cv2.findContours(threshold_GREEN,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Green: %d  "%len(contoursGREEN)
cv2.drawContours(FiltMedian_GREEN,contoursGREEN,-1,(0,255,255),2)
#cv2.namedWindow('Display2',cv2.WINDOW_NORMAL)

#Blue:
#result
res_BLUE = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_BLUE) 
FiltMedian_BLUE=cv2.medianBlur(res_BLUE,15)  # Filter, Smooth, Blur
#Count and print
gray_BLUE = cv2.cvtColor(FiltMedian_BLUE,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
res_BLUE = cv2.morphologyEx(gray_BLUE,cv2.MORPH_OPEN,kernel)

ret , threshold_BLUE = cv2.threshold(res_BLUE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursBLUE,_=cv2.findContours(threshold_BLUE,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Blue: %d  "%len(contoursBLUE)
cv2.drawContours(FiltMedian_BLUE,contoursBLUE,-1,(0,255,255),2)
#cv2.namedWindow('Display3',cv2.WINDOW_NORMAL)

#Yellow:
#result
res_YELLOW = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_YELLOW) 
FiltMedian_YELLOW=cv2.medianBlur(res_YELLOW,15)  # Filter, Smooth, Blur
#Count and print
gray_YELLOW = cv2.cvtColor(FiltMedian_YELLOW,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
res_YELLOW = cv2.morphologyEx(gray_YELLOW,cv2.MORPH_OPEN,kernel)

ret , threshold_YELLOW = cv2.threshold(res_YELLOW,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursYELLOW,_=cv2.findContours(threshold_YELLOW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Yellow: %d  "%len(contoursYELLOW)
cv2.drawContours(FiltMedian_YELLOW,contoursYELLOW,-1,(0,255,255),2)
#cv2.namedWindow('Display4',cv2.WINDOW_NORMAL)

#Black:
#result
res_BLACK = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_BLACK) 
FiltMedian_BLACK=cv2.medianBlur(res_BLACK,15)  # Filter, Smooth, Blur
#Count and print
gray_BLACK = cv2.cvtColor(FiltMedian_BLACK,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(90,90))
res_BLACK = cv2.morphologyEx(gray_BLACK,cv2.MORPH_OPEN,kernel)

ret , threshold_BLACK = cv2.threshold(res_BLACK,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursBLACK,_=cv2.findContours(threshold_BLACK,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Black: %d  "%len(contoursBLACK)
cv2.drawContours(FiltMedian_BLACK,contoursBLACK,-1,(0,255,255),2)
#cv2.namedWindow('Display1',cv2.WINDOW_NORMAL)

#Orange:
#result
res_ORANGE = cv2.bitwise_and(OrigImg,OrigImg, mask = mask_ORANGE) 
FiltMedian_ORANGE=cv2.medianBlur(res_ORANGE,15)  # Filter, Smooth, Blur
#Count and print
gray_ORANGE= cv2.cvtColor(FiltMedian_ORANGE,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(90,90))
res_ORANGE = cv2.morphologyEx(gray_ORANGE,cv2.MORPH_OPEN,kernel)

ret , threshold_ORANGE = cv2.threshold(res_ORANGE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contoursORANGE,_=cv2.findContours(threshold_ORANGE,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print
print "Number of Orange: %d  "%len(contoursORANGE)
cv2.drawContours(FiltMedian_ORANGE,contoursORANGE,-1,(0,255,255),2)
#cv2.namedWindow('Display1',cv2.WINDOW_NORMAL)



#Display images:
while True:
    #cv2.imshow("mask_RED",mask_RED) 
    #cv2.imshow("res_RED",res_RED)
    cv2.imshow("FMResImg_RED",FiltMedian_RED)
    cv2.imshow("FMResImg_BLUE",FiltMedian_BLUE)
    cv2.imshow("FMResImg_GREEN",FiltMedian_GREEN)
    cv2.imshow("FMResImg_YELLOW",FiltMedian_YELLOW)
    cv2.imshow("FMResImg_BLACK",FiltMedian_BLACK)
    cv2.imshow("FMResImg_ORANGE",FiltMedian_ORANGE)
    #cv2.imshow("OrigImg",OrigImg) # Displays the frame, mask and res in 3 separate windows.
    #cv2.imshow('Display',FiltMedian_RED)
    #cv2.imshow("Threshold_RED",threshold_RED)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


 



