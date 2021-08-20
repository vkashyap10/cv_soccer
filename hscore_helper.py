from return_hFeatureScore import return_hFeatureScore
import numpy as np
import matplotlib.pyplot as plt

def hscore_helper(frame1,grad_x,grad_y,hc):

    # print("hc in helper function before", hc)

    # find best point to track in the window
    window = 10
    imCrop = frame1[int(hc[0])-window:int(hc[0])+window,int(hc[1])-window:int(hc[1])+window]
    grad_xcrop = grad_x[int(hc[0])-window:int(hc[0])+window,int(hc[1])-window:int(hc[1])+window]
    grad_ycrop = grad_y[int(hc[0])-window:int(hc[0])+window,int(hc[1])-window:int(hc[1])+window]
    # find best feature to track within cropped image
    h_score = return_hFeatureScore(grad_xcrop,grad_ycrop)
    threshold = np.sort(h_score.ravel())[-1]
    hc_crop= np.where(h_score >= threshold)

    # get coordinates of best feature in full image
    hc_new = np.zeros((2,1))
    hc_new[0] = hc_crop[0] + int(hc[0])-window
    hc_new[1] = hc_crop[1] + int(hc[1])-window


    # plt.imshow(imCrop)
    # plt.scatter(hc_crop[1], hc_crop[0], marker="x", color="red", s=200)
    # plt.title("Best harris corner in full image")
    # plt.show()

    print("diff hc in helper ", hc_new - hc)

    return hc_new
