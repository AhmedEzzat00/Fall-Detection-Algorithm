# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:03:40 2019

@author: Ahmed
"""

import numpy as np
import cv2
import libbgs
'''
x= have all problems (Excluded)
ff= first frame
s&p= salt and pepper
bc= broken contours
inv= when the subject hold still becomes INVISIBLE
cs =continuos shadow
'''
def area_of_contour(cnt):
    area = cv2.contourArea(cnt)
    return area
# Detect parameters
minArea = 80*80 
thresholdLimit = 20

## BGS Library algorithms
#bgs = libbgs.FrameDifference()              #inv #bc
#bgs = libbgs.StaticFrameDifference()         #ff #s&p #bc
#bgs = libbgs.AdaptiveBackgroundLearning()     #cs #inv #s&p
#bgs = libbgs.AdaptiveSelectiveBackgroundLearning()   #x
#bgs = libbgs.DPAdaptiveMedian()               #bc #ff
#bgs = libbgs.DPEigenbackground()               #x
#bgs = libbgs.DPGrimsonGMM()                #x
#bgs = libbgs.DPMean()                      #bc #inv
#bgs = libbgs.DPPratiMediod()               #x
#bgs = libbgs.DPTexture()                   #x
#bgs = libbgs.DPWrenGA()                    #x
#bgs = libbgs.DPZivkovicAGMM()              #x
#bgs = libbgs.FuzzyChoquetIntegral()         #x
#bgs = libbgs.FuzzySugenoIntegral()          #x
#bgs = libbgs.GMG() # if opencv 2.x
#bgs = libbgs.IndependentMultimodal()
#bgs = libbgs.KDE()                          x
#bgs = libbgs.KNN() # if opencv 3.x
#bgs = libbgs.LBAdaptiveSOM()
#bgs = libbgs.LBFuzzyAdaptiveSOM()
#bgs = libbgs.LBFuzzyGaussian()
#bgs = libbgs.LBMixtureOfGaussians()
#bgs = libbgs.LBSimpleGaussian()
#bgs = libbgs.LBP_MRF()
#bgs = libbgs.LOBSTER()
#bgs = libbgs.MixtureOfGaussianV1() # if opencv 2.x
#bgs = libbgs.MixtureOfGaussianV2()
#bgs = libbgs.MultiCue()
#bgs = libbgs.MultiLayer()
#bgs = libbgs.PAWCS()
##bgs = libbgs.PixelBasedAdaptiveSegmenter()
#bgs = libbgs.SigmaDelta()
#bgs = libbgs.SuBSENSE()
#bgs = libbgs.T2FGMM_UM()
#bgs = libbgs.T2FGMM_UV()
#bgs = libbgs.T2FMRF_UM()
#bgs = libbgs.T2FMRF_UV()
#bgs = libbgs.VuMeter()
#bgs = libbgs.WeightedMovingMean()
#bgs = libbgs.WeightedMovingVariance()
#bgs = libbgs.TwoPoints()
#bgs = libbgs.ViBe()
##bgs = libbgs.CodeBook()


#video_file = "Background Subtraction Algorthim/dataset/Coffee_room_02/Videos/video (52).avi"

video_file = "dataset/Coffee_room_02/Videos/video (52).avi"

capture = cv2.VideoCapture(video_file)

  
while not capture.isOpened():
	capture = cv2.VideoCapture(video_file)
	cv2.waitKey(1000)
	print "Wait for the header"

pos_frame = capture.get(1)
while True:
	flag, frame = capture.read()
	
	if flag:
            cv2.imshow('video', frame)
            pos_frame = capture.get(1)
            img_output = bgs.apply(frame)
            img_bgmodel = bgs.getBackgroundModel();
            gray_img=cv2.cvtColor(img_bgmodel,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray',gray_img)
            thresh = cv2.threshold(gray_img, thresholdLimit, 255, cv2.THRESH_BINARY)[1]
            _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #find contours
            contours=sorted(contours,key=area_of_contour ,reverse=False) ## contours are sorted in ascending order, largest is at the end
             
            if len(contours)!=0:            
	            selected_cnt=contours[-1]     ## here i take the last contour in the array which is the largest 
            else :
                continue
            
            if area_of_contour(selected_cnt)<minArea:
                print(area_of_contour(selected_cnt))
                continue
            
            if len(selected_cnt)>=5: 
               
                (x, y, w, h) = cv2.boundingRect(selected_cnt)
                ellipse = cv2.fitEllipse(selected_cnt)
                rect = cv2.minAreaRect(selected_cnt)
                algorismContourAngle=rect[2]*(-1)  ## this is the angle retuned by minAreaRect algorism (normally width is less than height)
                myContourAngle=algorismContourAngle + 90 ## this is the corrected angle if width is less than height
                box = cv2.boxPoints(rect)  ## points to draw the rotated rectangle
                box = np.int0(box)
                # drawing bounding rectangle, ellipse, and minimun rotated rectangle  on the current frame
                cv2.ellipse(img_bgmodel,ellipse,(255,0,0),2)
                cv2.rectangle(img_bgmodel, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(img_bgmodel,[box],0,(100,255,0),2)
                
                aspect_ratio = float(w)/h
                #angle of ellipse with vertical and horizontal 
                angle_ellipse_with_vertical= ellipse[2]
                if ellipse[2]<90: 
                        angle_ellipse_with_horizontal=90-ellipse[2]
                else:
                        angle_ellipse_with_horizontal=(ellipse[2]-90)*(-1)
                ## rotated rectangle angle of large axis with horizontal        
                if rect[1][0]<rect[1][1]:        #if width of rotated rectangle<its length , then algorism will find the angle between horizontal and the first edge it encounters anticolcokwise(in this case the smaller axis) we want the anngle of horizontal with the longer axis and not the smaller axis
                         rotrect_angle= myContourAngle 
                else:
                         rotrect_angle=algorismContourAngle
                print("AspetR " ,aspect_ratio)
                print("angle " ,rotrect_angle)
	                     
            cv2.imshow('img_output', img_output)
            cv2.imshow('img_bgmodel', img_bgmodel)
           
	if 0xFF & cv2.waitKey(10) == 27:
		break

cv2.destroyAllWindows()
