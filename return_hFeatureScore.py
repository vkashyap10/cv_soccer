import numpy as np
import cv2
from numpy import linalg as LA

# harris corner response function

def return_hFeatureScore(grad_x,grad_y):

    h_score =np.zeros(grad_x.shape)
    window_list = np.arange(5) - 2

    # alpha, used in harris response function
    alpha = 0.05

    border = int(len(window_list)/2 + 1)
    # initialize the moment matrix
    for i in range(border,grad_x.shape[0]-border,5):
        for j in range(border,grad_y.shape[1]-border,5):
            M = np.zeros((2,2))
            for k in window_list:
                for l in window_list:
                    M[0,0] = M[0,0] + (grad_x[i+k,j+l])**2
                    M[1,1] = M[1,1] + (grad_y[i+k,j+l])**2
                    M[1,0] = M[1,0] + grad_y[i+k,j+l]*grad_x[i+k,j+l]
                    M[0,1] = M[0,1] + grad_y[i+k,j+l]*grad_x[i+k,j+l]

            # get eigen-values
            w, v = LA.eig(M)
            h_score[i,j] = w[0]*w[1] - alpha*((w[0] + w[1])**2)

    return h_score