
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA

def klt_tracker(frame0,frame1,gradx,grady,f2trackcord):

	#solve for u,v
	# get a 3X3 matrix around the feature to track and write lucas kanade equations
	frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

	window = 20
	frame2track0 = frame0[int(f2trackcord[0]-window):int(f2trackcord[0]+window+1),int(f2trackcord[1]-window):int(f2trackcord[1]+window+1)]
	gradx_frame = gradx[int(f2trackcord[0]-window):int(f2trackcord[0]+window+1),int(f2trackcord[1]-window):int(f2trackcord[1]+window+1)]
	grady_frame = grady[int(f2trackcord[0]-window):int(f2trackcord[0]+window+1),int(f2trackcord[1]-window):int(f2trackcord[1]+window+1)]

	# temporal difference
	frame2track1 = frame1[int(f2trackcord[0]-window):int(f2trackcord[0]+window+1),int(f2trackcord[1]-window):int(f2trackcord[1]+window+1)]
	It = frame2track1 - frame2track0

	Atb = np.zeros((2,1))

	# moment matrix
	M = np.zeros((2,2))
	for i in range(gradx_frame.shape[0]):
		for j in range(gradx_frame.shape[1]):
		    M[0,0] = M[0,0] + (gradx_frame[i,j])**2
		    M[1,1] = M[1,1] + (grady_frame[i,j])**2
		    M[1,0] = M[1,0] + grady_frame[i,j]*gradx_frame[i,j]
		    M[0,1] = M[0,1] + grady_frame[i,j]*gradx_frame[i,j]

		    Atb[0,0] = Atb[0,0] + It[i,j]*gradx_frame[i,j]
		    Atb[1,0] = Atb[1,0] + It[i,j]*grady_frame[i,j]

	vel = np.matmul(np.linalg.inv(M),-Atb)

	# given velocity find feature in new frame
	# navie : assume tracking is optimal
	f2trackcord_new = f2trackcord + (vel).astype(int)

	# find best new match
	# best_match = f2trackcord_new
	# ssd_min = 1e9
	# for i in range(20):
	# 	for j in range(20):
	# 		newc_i = f2trackcord_new[0] - i + 10
	# 		newc_j = f2trackcord_new[1] - j + 10
	# 		new_window = frame1[int(newc_i-window):int(newc_i+window+1),int(newc_j-window):int(newc_j+window+1)]
	# 		ssd = np.sum((new_window- frame2track0)**2)
	# 		if(ssd < ssd_min):
	# 			ssd_min = ssd
	# 			best_match[0] = newc_i
	# 			best_match[1] = newc_j

	print("best_match",f2trackcord_new)

	return f2trackcord_new,(vel).astype(int)
