# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:46:23 2019

@author: Era
"""

import cv2

#importing the video and finding the numbers of frames included in it
cap = cv2.VideoCapture ('/trafficVideo.mp4')
# performing background subtraction using MOG2
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
	#finding countours using morphological open to create bounding box
	#convert a binary mask to a bounding box by finding the border pixels of the masked image
    ret,th1 = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#showing the masked result
    
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                #geting the 4 points of rectangle
                x, y, w, h = cv2.boundingRect(contours[i])
                
                if (w*h>1500):
                    #Creating rectangle and ellipse
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # showing the original video
                    cv2.imshow('frame2',frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()