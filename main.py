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

    # select bounding box
    bbox = cv2.selectROI(frame)

    # Crop image
    imCrop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # compute gradient of image
    grad_x,grad_y = return_gradient(imCrop)

    # verify goal post corners are harris corners by computing harris conrers for cropped image
    h_score = return_hFeatureScore(grad_x,grad_y)
    # threshold = np.nanpercentile(h_score,99)
    threshold = np.sort(h_score.ravel())[-10]

    h_corners = np.where(h_score > threshold)
    print("number of harris corners detected ", len(h_corners[0]))

    plt.imshow(imCrop)
    plt.scatter(h_corners[1], h_corners[0], marker="x", color="red", s=200)
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
