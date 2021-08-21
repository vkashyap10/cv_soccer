import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
from return_gradient import return_gradient
from return_hFeatureScore import return_hFeatureScore
from klt_tracker import klt_tracker
from hscore_helper import hscore_helper
from getCorners2Track import getCorners2Track
from h_main import solveHomographySvd
from h_main import inverse_warp

cap = cv2.VideoCapture('data/project.mp4')
img2warp = cv2.imread('data/golden.jpg')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (35,35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
print("frame type", old_frame.dtype)

hc = getCorners2Track(old_frame)

plt.imshow(old_frame)
plt.scatter(hc[:,:,0], hc[:,:,1], marker="x", color="red", s=200)
plt.title("Best harris corner")
plt.show()

# store initial points to track 
p0 = hc.copy()
p0 = np.float32(p0)
print("p0 shape")
print(p0.shape)

# pseudo world points for computing homography
p_old = np.zeros(p0.shape)
p_old[0,:,:] = 0,old_frame.shape[0]
p_old[2,:,:] = old_frame.shape[1],0
p_old[3,:,:] = old_frame.shape[1],old_frame.shape[0]

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
list_frames = []
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # calculate homographic transformation
    m = solveHomographySvd(p_old,p1).reshape(3,3)
    new_image = inverse_warp(img2warp,m)
    new_image = np.uint8(new_image)
    frame = cv2.add(frame,new_image)
    
    # frame = ((frame + new_image)/2).astype(int)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    list_frames.append(img)
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()