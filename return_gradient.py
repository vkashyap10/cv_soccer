import numpy as np
import cv2
from numpy import linalg as LA

# function to return gradient in an image using sobel operator

def return_gradient(imCrop):
    # convert image to greyscale
    grayscale_im = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)

    # store gradient
    gradient_im_x = np.zeros(grayscale_im.shape)
    gradient_im_y = np.zeros(grayscale_im.shape)
    # apply Sobel operator
    for i in range(1,grayscale_im.shape[0]-1):
        for j in range(1,grayscale_im.shape[1]-1):
            gradient_im_x[i,j] = (2/8*grayscale_im[i,j+1] - 2/8*grayscale_im[i,j-1]) 
            + 1/8*grayscale_im[i+1,j+1]+ 1/8*grayscale_im[i-1,j+1] - (1/8*grayscale_im[i-1,j-1]+ 1/8*grayscale_im[i+1,j-1] )

    for i in range(1,grayscale_im.shape[0]-1):
        for j in range(1,grayscale_im.shape[1]-1):
            gradient_im_y[i,j] = (2/8*grayscale_im[i+1,j] - 2/8*grayscale_im[i-1,j]) 
            + 1/8*grayscale_im[i+1,j+1]+ 1/8*grayscale_im[i+1,j-1] - (1/8*grayscale_im[i-1,j-1]+ 1/8*grayscale_im[i-1,j+1] )

    return gradient_im_x,gradient_im_y