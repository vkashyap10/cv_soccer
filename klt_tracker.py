
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA

def klt_tracker(frame0,frame1,gradx,grady,f2trackcord):

	#solve for u,v
	# get a 3X3 matrix around the feature to track and write lucas kanade equations
	frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

	frame2track0 = frame0[int(f2trackcord[0]-1):int(f2trackcord[0]+2),int(f2trackcord[1]-1):int(f2trackcord[1]+2)]
	gradx_frame = gradx[int(f2trackcord[0]-1):int(f2trackcord[0]+2),int(f2trackcord[1]-1):int(f2trackcord[1]+2)]
	grady_frame = grady[int(f2trackcord[0]-1):int(f2trackcord[0]+2),int(f2trackcord[1]-1):int(f2trackcord[1]+2)]

	# temporal difference
	frame2track1 = frame1[int(f2trackcord[0]-1):int(f2trackcord[0]+2),int(f2trackcord[1]-1):int(f2trackcord[1]+2)]
	It = frame2track1 - frame2track0

	Atb = np.zeros((2,1))

	# moment matrix
	M = np.zeros((2,2))
	for i in range(3):
		for j in range(3):
		    M[0,0] = M[0,0] + (gradx_frame[i,j])**2
		    M[1,1] = M[1,1] + (grady_frame[i,j])**2
		    M[1,0] = M[1,0] + grady_frame[i,j]*gradx_frame[i,j]
		    M[0,1] = M[0,1] + grady_frame[i,j]*gradx_frame[i,j]

		    print()
		    Atb[0,0] = Atb[0,0] + It[i,j]*gradx_frame[i,j]
		    Atb[1,0] = Atb[1,0] + It[i,j]*grady_frame[i,j]

	vel = np.matmul(np.linalg.inv(M),-Atb)

	return vel
