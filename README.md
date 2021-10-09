# Introduction

You must have seen a football match screened in multiple countires shows different advertisements on the board. This is an attempt to demistify the working of that technology. 

# To run
Clone the repo and run main.py
While selecting Harris corneres in your region of interest, start with the bottom left and move in an anticlockwise manner. 

Saved solution video file : track_homography.mp4

Original video : data/ronaldo_goal.mp4

# How it works
1) Harris corners in the image are detected. The four corners of the goal post come out as good harris corners.

![image](https://user-images.githubusercontent.com/31934929/136610404-c578a6a0-513c-4299-9c65-479170254611.png)

2) After detecting the corners use local constraints from Lucas Kanade to solve the aperature problem (use Linear regression). This will give velocity of all four harris corners.

3) Once we have 4 corners in the new location, solve homography problem using Singular Value Decomposition.

4) After gettting the solution, project the desired image onto the frame using inverse warping.

5) Do this for all frames and save the video.

# How the solution looks.

<img width="1162" alt="Screenshot 2021-10-08 at 7 54 53 PM" src="https://user-images.githubusercontent.com/31934929/136611362-0e9a1a2e-d3f0-463e-b020-51bb249ef489.png">


Future steps:
Combine Optical Flow detection using Lucas Kanade with Kalman filter to allow tracking of data points through obstruction.

