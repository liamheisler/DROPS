'''
Senior Design, Group MEM 23 Initial Testing Scripts
Group: L. Heisler, J. Bunch, C. Silvia, R. Zegarski, J. Heinzman
'''

__author__ = "Liam Heisler, MEM 23"
__version__ = "x.x"
__status__ = "Development"

# Computer vision imports
import cv2
#from picamera import PiCamera
#from picamera.array import PiRGBArray

# Utility Imports
import numpy as np, imutils
from datetime import datetime
from imutils import perspective
from imutils import contours

#from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# ADJUSTABLE PARAMETERS
blur = 7 # smoothness between background/foreground
canny_low = 15 # min intensity for edge definition
canny_high = 125 # max "" 
dilate_iter = 0 # num of iterations of dilation
erode_iter = 0 # "" of erosion
min_area = 0.0002 # % of total area a contour can occupy
max_area = 0.05 # % ""
epsilon_pct_arc = 0.02 # threshold for the approxPolyDP function (contour curve est.)
#mask_color = (0.0, 0.0, 0.0) # color of background once its removed (black)

numCorners = 5 # how many sides on the face we are looking at?

# contour checker
def check_contour(cnt, length, numCorners, epsilon_pct_arc, img_area, min_area_perct, max_area_perct):
    approx = cv2.approxPolyDP(cnt, epsilon_pct_arc*length, True)

    cnt_area = cv2.contourArea(cnt)
    min_area = min_area_perct * img_area
    max_area = max_area_perct * img_area

    # yields FALSE if the contour does not have numCorners sides
    if cnt_area > min_area and cnt_area < max_area:
        print("img: {}, min: {}, max: {}, cnt: {}".format(img_area, min_area, max_area, cnt_area))
        if len(approx) == numCorners:
            # valid contour (size & shape)
            return True
        else:
            # invalid contour (size & shape)
            return False
    else:
        # invalid contour (size)
        return False
    
while True:
    ret, frame, = cap.read()
    if ret: # we have a feed
        img_area = frame.shape[0] * frame.shape[1]

        # init mask to remove the non-5-sided objects
        # shape[:2] yields resolution of img
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

        # Apply edge detection
        edged = cv2.Canny(gray, canny_low, canny_high)

        # Neccessary? Dilated and eroded to ensure gaps are closed

        if dilate_iter != 0 and erode_iter != 0:
            edged = cv2.dilate(edged, None, iterations=dilate_iter)
            edged = cv2.erode(edged, None, iterations=erode_iter)
        else:
            edged = edged

        # Find contours in the image (enclosed areas)
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # filter contours from based on validity
        numValid = 0
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            contour_valid = check_contour(cnt, peri, numCorners, epsilon_pct_arc, img_area, min_area, max_area)
            if contour_valid:
                # overlay valid contour on the image frame
                cv2.drawContours(frame, [cnt], 0, 255, -1)
                numValid += 1
            elif not contour_valid:
                # add contour to the mask (to be removed)
                cv2.drawContours(mask, [cnt], -1, 0, -1)         
            
        #print(numValid)
        edged_masked = cv2.bitwise_and(edged, edged, mask=mask)
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Edged BEFORE Mask Application", edged)
        cv2.imshow("Edged AFTER Mask Application", edged_masked)
        
        key = cv2.waitKey(1)
        
        if key == 115: # "s" key
            break
        if key == 99: # "c" key
            now = datetime.now() # current data and time
            now = now.strftime("%m%d%Y_%H%M%S")
            img_name = "capture_" + now + ".png"
            cv2.imwrite(img_name, edged_masked)
            print("img written: " + img_name)
    
cap.release()
cv2.destroyAllWindows()

