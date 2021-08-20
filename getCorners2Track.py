import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from return_gradient import return_gradient
from return_hFeatureScore import return_hFeatureScore
from klt_tracker import klt_tracker
from hscore_helper import hscore_helper

def getCorners2Track(old_frame):
    # get harris corners from desired cropped image
    # select bounding box
    points2get = 4
    hc = np.zeros((points2get,1,2))

    for pt_inx in range(points2get):
        bbox = cv2.selectROI(old_frame)

        # compute gradient of crop image
        grad_x,grad_y = return_gradient(old_frame,int(bbox[1]),int(bbox[1]+bbox[3]), int(bbox[0]),int(bbox[0]+bbox[2]))

        # Crop image and gradients
        imCrop = old_frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        grad_xcrop = grad_x[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        grad_ycrop = grad_y[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

        # find best feature to track within cropped image
        num_features = 1
        h_score = return_hFeatureScore(grad_xcrop,grad_ycrop)
        threshold = np.sort(h_score.ravel())[-num_features]
        hc_crop= np.array(np.where(h_score >= threshold))

        # get coordinates of best feature in full image
        hc[pt_inx,:,1] = (hc_crop[1,:] + int(bbox[1]))[:,np.newaxis]
        hc[pt_inx,:,0] = (hc_crop[0,:] + int(bbox[0]))[:,np.newaxis]

    return hc

    