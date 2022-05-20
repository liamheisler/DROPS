'''
Senior Design, Group MEM 23 Initial Testing Scripts
Group: L. Heisler, J. Bunch, C. Silvia, R. Zegarski, J. Heinzman
'''

__author__ = "Liam Heisler, MEM 23"
__version__ = "x.x"
__status__ = "Development"

# Utility Imports
import cv2
import numpy as np
import glob
import imutils
from datetime import datetime

# Define cam and some basic settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camW = 1500 # camera pixel width
camH = 900 # camera pixel height
#camRate = 40 # camera's frame rate

# Set camera config parameters
use = True
if use:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camH)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn off the AutoFocus
    #cap.set(5, camRate)

use_live_cap = False

while use_live_cap:
    ret, frame, = cap.read()
    #print(frame.shape)

    if ret: # we have a feed
        cv2.imshow("chessboard capturing...", frame)
        img_area = frame.shape[0] * frame.shape[1]

    key = cv2.waitKey(1)
    if key == 115: # "s" key
        break
    if key == 99: # "c" key
        now = datetime.now() # current data and time
        now = now.strftime("%m%d%Y_%H%M%S")
        img_name = "chessboard_" + now + ".png"
        cv2.imwrite(img_name, frame)
        print("chessboard written: " + img_name)
        
use_calibration = True
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if use_calibration:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("-- INTRINSIC MATRIX -- ")
    print(mtx)
    np.save("usb_mtx", mtx)
    print("-- DISTORTION COEFF -- ")
    print(dist)
    np.save("usb_dist", dist)


cap.release()
cv2.destroyAllWindows()


