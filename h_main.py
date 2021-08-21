import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from return_gradient import return_gradient
from return_hFeatureScore import return_hFeatureScore
from klt_tracker import klt_tracker
from hscore_helper import hscore_helper
from getCorners2Track import getCorners2Track

def get_xrow(x,y,x_new,y_new):
	return np.array([-x,-y,-1,0,0,0,x*x_new,y*x_new,x_new])

def get_yrow(x,y,x_new,y_new):
	return np.array([0,0,0,-x,-y,-1,x*y_new,y*y_new,y_new])

def inverse_warp(old_image,m):
	new_image = np.zeros(old_image.shape)
	cord = []
	for i in range(old_image.shape[0]):
		for j in range(old_image.shape[1]):
			cord.append(np.array([i,j,1]))

	cord = np.array(cord)
	prev_cord = np.matmul(np.linalg.inv(m),cord.T)
	prev_cord = (prev_cord/prev_cord[2,:])[:2,:]
	prev_cord = np.rint(prev_cord)

	ctr = -1
	for i in range(old_image.shape[0]):
		for j in range(old_image.shape[1]):
			ctr = ctr + 1
			if(int(prev_cord[0,ctr]) < 0 or int(prev_cord[0,ctr]) >= old_image.shape[0]):
				continue

			if(int(prev_cord[1,ctr]) < 0 or int(prev_cord[1,ctr]) >= old_image.shape[1]):
				continue

			new_image[i,j,:] = old_image[int(prev_cord[0,ctr]),int(prev_cord[1,ctr]),:]

	return new_image

def solveHomographySvd(p_old,p0):
	A = []
	for i in range(p0.shape[0]):
		A.append(get_xrow(p_old[i,:,1],p_old[i,:,0],p0[i,:,1],p0[i,:,0]))
		A.append(get_yrow(p_old[i,:,1],p_old[i,:,0],p0[i,:,1],p0[i,:,0]))

	A = np.array(A)

	AtA = np.matmul((A/1e3).T,A/1e3)
	AtA = np.array(AtA, dtype=float)

	w,v = np.linalg.eig(AtA)
	min_index = np.argmin(w)

	m = v[:,min_index]

	return m
