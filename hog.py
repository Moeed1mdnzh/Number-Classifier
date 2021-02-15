import cv2 
import numpy as np

winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivA = 1
winSigma = -1 
histNT = 0 
l2 = 0.2
gamma = True 
nlevels = 64 
Sgradient = True
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
	derivA,winSigma,histNT,l2,gamma,nlevels,Sgradient)
hog.save("HOG.xml")