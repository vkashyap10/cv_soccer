import numpy as np
import cv2
from numpy import linalg as LA

# function to return gradient in an image using sobel operator

def return_gradient(imCrop,sind_x,eind_x,sind_y,eind_y):
    # convert image to greyscale
    grayscale_im = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)

    # store gradient
    gradient_im_x = np.zeros(grayscale_im.shape)
    gradient_im_y = np.zeros(grayscale_im.shape)
    # apply Sobel operator
    for i in range(sind_x,eind_x):
        for j in range(sind_y,eind_y):
            gradient_im_x[i,j] = (2/8*grayscale_im[i,j+1] - 2/8*grayscale_im[i,j-1]) 
            + 1/8*grayscale_im[i+1,j+1]+ 1/8*grayscale_im[i-1,j+1] - (1/8*grayscale_im[i-1,j-1]+ 1/8*grayscale_im[i+1,j-1] )

    for i in range(sind_x,eind_x):
        for j in range(sind_y,eind_y):
            gradient_im_y[i,j] = (2/8*grayscale_im[i+1,j] - 2/8*grayscale_im[i-1,j]) 
            + 1/8*grayscale_im[i+1,j+1]+ 1/8*grayscale_im[i+1,j-1] - (1/8*grayscale_im[i-1,j-1]+ 1/8*grayscale_im[i-1,j+1] )

    return gradient_im_x,gradient_im_y