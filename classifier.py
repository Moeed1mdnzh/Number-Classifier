import cv2 
import numpy as np 


board = np.zeros((500,600),np.uint8)
drawPerm = False
svm = cv2.ml.SVM_load("SVM.xml")
res = ""
def nothing(x):
	pass

def Deskewed(image):
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

def pen(event,x,y,flags,param):
	global board,penSize,drawPerm
	if event == cv2.EVENT_LBUTTONDOWN:
		drawPerm = True 
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawPerm:
			cv2.circle(board,(x,y),penSize,255,-1)
	elif event == cv2.EVENT_LBUTTONUP:
		drawPerm = False
cv2.namedWindow("Board")
cv2.createTrackbar("RESET","Board",0,1,nothing)
cv2.createTrackbar("RESPOND","Board",0,1,nothing)
cv2.createTrackbar("penSize","Board",5,100,nothing)
cv2.setMouseCallback("Board",pen)
while True:
	reset_perm = cv2.getTrackbarPos("RESET","Board")
	respond_perm = cv2.getTrackbarPos("RESPOND","Board")
	penSize = cv2.getTrackbarPos("penSize","Board")
	if reset_perm:
		cv2.setTrackbarPos("RESET","Board",0)
		cv2.setTrackbarPos("penSize","Board",5)
		board = np.zeros((500,600),np.uint8)
	clone = board.copy()
	if respond_perm:
		cv2.setTrackbarPos("RESPOND","Board",0)
		resized = cv2.resize(board,(20,20),interpolation=cv2.INTER_AREA)
		test_deskewed = Deskewed(resized)
		test_hog = HOG_desc(test_deskewed)
		data = np.float32(test_hog).reshape(-1,81)
		res = str(int(svm.predict(data)[1]))
	cv2.putText(clone,f"Classified:{res}",(20,50),cv2.FONT_HERSHEY_TRIPLEX,
		1,255,2)
	cv2.imshow("Board",clone)
	if cv2.waitKey(1) == ord("q"):
		quit()