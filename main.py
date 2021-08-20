import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from return_gradient import return_gradient
from return_hFeatureScore import return_hFeatureScore
from klt_tracker import klt_tracker
from hscore_helper import hscore_helper

if __name__ == "__main__": 

    # Read video
    video = cv2.VideoCapture("data/project.mp4")
    print("is video open successful", video.isOpened())

    # Read first frame.
    ok, frame0 = video.read()
    
    # select bounding box
    bbox = cv2.selectROI(frame0)

    # compute gradient of crop image
    grad_x,grad_y = return_gradient(frame0,int(bbox[1]),int(bbox[1]+bbox[3]), int(bbox[0]),int(bbox[0]+bbox[2]))

    # Crop image and gradients
    imCrop = frame0[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    grad_xcrop = grad_x[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    grad_ycrop = grad_y[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # find best feature to track within cropped image
    h_score = return_hFeatureScore(grad_xcrop,grad_ycrop)
    threshold = np.sort(h_score.ravel())[-1]
    hc_crop= np.where(h_score >= threshold)

    # get coordinates of best feature in full image
    hc = np.zeros((2,1))
    hc[0] = hc_crop[0] + int(bbox[1])
    hc[1] = hc_crop[1] + int(bbox[0])
    plt.imshow(frame0)
    plt.scatter(hc[1], hc[0], marker="x", color="red", s=200)
    plt.title("Best harris corner")
    plt.show()

    # store frames to make video
    frames_list = []
    frame0_temp = frame0.copy()
    frame0_temp[int(hc[0]),int(hc[1]),:] = [0,0,255]
    frames_list.append(frame0_temp)

    vel_tracker_x = []
    vel_tracker_y = []

    vel_correction_x = []
    vel_correction_y = []

    ctr = 0
    while(1):
        ok, frame1 = video.read()
        try:
            # track the best corner
            hc,vel = klt_tracker(frame0,frame1,grad_x,grad_y,hc)
            vel_tracker_x.append(vel[0])
            vel_tracker_y.append(vel[1])
        except:
            break

        # compute new gradient
        bbox_self = 30
        grad_x,grad_y = return_gradient(frame1,int(hc[0]-bbox_self), int(hc[0]+bbox_self),int(hc[1]-bbox_self), int(hc[1]+bbox_self))

        # find best point to track in the window
        # if ctr%5 == 0:
        hc_old = hc.copy()
        hc = hscore_helper(frame1,grad_x,grad_y,hc)
        vel_correction_x.append(hc[0] - hc_old[0])
        vel_correction_y.append(hc[1] - hc_old[1])

        frame0 = frame1.copy()
        frame1[int(hc[0])-3:int(hc[0])+3,int(hc[1])-3:int(hc[1])+3,:] = [0,0,255]
        frames_list.append(frame1)


    print("total frames processed", len(frames_list))

    # save video
    height, width, layers = frames_list[0].shape
    size = (width,height)
    out = cv2.VideoWriter('track.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
     
    for i in range(len(frames_list)):
        out.write(frames_list[i])
    out.release()


    plt.plot(vel_tracker_x,label='velocity due to KL tracker')
    plt.plot(vel_correction_x,label = 'velocity due to harris correction')
    plt.title("velocity in x direction")
    plt.legend()
    plt.show()

    # Display gradients image
    # cv2.imshow("grayscale_im", grayscale_im)

    # plt.imshow(grad_x,cmap=plt.cm.gray)
    # plt.title("gradient in x")
    # plt.show()

    # plt.imshow(grad_y,cmap=plt.cm.gray)
    # plt.title("gradient in y")
    # plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
