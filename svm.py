import cv2 
from cv2 import ml 
import numpy as np 

def Deskewd(image):
	SZ = 20 
	affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
	m = cv2.moments(image)
	if abs(m["mu02"] < 0.01):
		return image.copy() 
	else:
		skew = m["mu11"] / m["mu02"]
		M = np.float32([[1,skew,(-0.5*SZ*skew)],[0,1,0]])
		img = cv2.warpAffine(image,M,(SZ,SZ),flags=affine_flags)
	return img 

def HOG_desc(image):
	hog = cv2.HOGDescriptor("HOG.xml")
	des = hog.compute(image)
	return des 

samples = cv2.imread("digits.png",0)
samples = [np.hsplit(row,100) for row in np.vsplit(samples,50)]
train_des = [map(Deskewd,row) for row in samples]
train_hog = [list(map(HOG_desc,row)) for row in train_des]
data = np.float32(train_hog).reshape(-1,81)
k = np.arange(10)
train_resp = np.repeat(k,500)[:,np.newaxis]
svm = ml.SVM_create()
svm.setType(ml.SVM_C_SVC)
svm.setKernel(ml.SVM_RBF)
svm.setC(12.5)
svm.setGamma(0.50625)
svm.train(data,ml.ROW_SAMPLE,train_resp)
svm.save("SVM.xml")