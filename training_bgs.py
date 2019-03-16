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

## BGS Library algorithms
bgs = libbgs.FrameDifference()              #inv #bc
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
		
		cv2.imshow('img_output', img_output)
		cv2.imshow('img_bgmodel', img_bgmodel)

	else:
		cv2.waitKey(1000)
		break
	
	if 0xFF & cv2.waitKey(10) == 27:
		break

cv2.destroyAllWindows()
