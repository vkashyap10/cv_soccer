import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from return_gradient import return_gradient
from return_hFeatureScore import return_hFeatureScore


if __name__ == "__main__": 

    # Read video
    video = cv2.VideoCapture("data/project.mp4")
    print("is video open successful", video.isOpened())

    # Read first frame.
    ok, frame = video.read()
    print(frame.shape)
    # compute gradient of entire image
    grad_x,grad_y = return_gradient(frame)

    # select bounding box
    bbox = cv2.selectROI(frame)

    print("output from bbox", bbox)

    # Crop image and gradients
    imCrop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    grad_xcrop = grad_x[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    grad_ycrop = grad_y[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # find best feature to track within cropped image
    h_score = return_hFeatureScore(grad_xcrop,grad_ycrop)
    threshold = np.sort(h_score.ravel())[-1]
    hc_crop= np.where(h_score >= threshold)
    print("best harris corner at ", hc_crop)

    plt.imshow(imCrop)
    plt.scatter(hc_crop[1], hc_crop[0], marker="x", color="red", s=200)
    plt.title("Best harris coerner in cropped image")
    plt.show()

    # get coordinates of best feature in full image
    hc = np.zeros((2,1))
    hc[0] = hc_crop[0] + int(bbox[1])
    hc[1] = hc_crop[1] + int(bbox[0])
    plt.imshow(frame)
    plt.scatter(hc[1], hc[0], marker="x", color="red", s=200)
    plt.title("Best harris coerner in full image")
    plt.show()

    print("coordinates of best feature to track", hc)
    # klt_tracker(frame0,frame1,grad_x,grad_y,hc)

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
