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
area_peri_ratio_check = 0.9 
blur = 5 # smoothness between background/foreground
canny_low = 35 # min intensity for edge definition
canny_high = 125 # max ""
dilate_iter = 0 # num of iterations of dilation
erode_iter = 0 # "" of erosion
min_area = 0.000225 # % of total area a contour can occupy
max_area = 0.05 # % ""
epsilon_pct_arc = 0.02 # threshold for the approxPolyDP function (contour curve est.)
#mask_color = (0.0, 0.0, 0.0) # color of background once its removed (black)

numCorners = 5 # how many sides on the face we are looking at?

# Corner Detection Parameters
block_size = 2 # Size of neighborhood considered for corner detection
k_size = 3 # Aperature parameter of Sobel derivative used
harris_k = 0.04 # Haris detector free parameter
harris_confidence_pct = 0.15 # % of the max confidence that corner IS a corner

# contour checker
def check_contour(cnt, length, numCorners, epsilon_pct_arc, img_area, min_area_perct, max_area_perct):
    approx = cv2.approxPolyDP(cnt, epsilon_pct_arc*length, True)

    cnt_area = cv2.contourArea(cnt)
    min_area = min_area_perct * img_area
    max_area = max_area_perct * img_area

    # yields FALSE if the contour does not have numCorners sides
    if cnt_area > min_area and cnt_area < max_area:
        if (cnt_area/length) > area_peri_ratio_check: 
            if len(approx) == numCorners:
                print("len: {} img: {}, min: {}, max: {}, cnt: {}".format(len(approx),
                                                                          img_area,
                                                                          min_area,
                                                                          max_area,
                                                                          cnt_area))
                # valid contour (size & shape)
                return True
            else:
                return False
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

        if len(np.nonzero(edged_masked)[1]) == 0: # is the masked img empty?
            frame = frame # if so, keep as is
        else:
            dst = cv2.cornerHarris(edged_masked, block_size, k_size, harris_k) # params
            dst = cv2.dilate(dst, None)
            # if not, detect & set corners to red in the img frame
            frame[dst>harris_confidence_pct*dst.max()]=[0,0,255]

            # Gather the coordinates of the corners
            ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

            minX = min(corners[:,0])
            minY = min(corners[:,1])

            maxX = max(corners[:,0])
            maxY = max(corners[:,1])
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Edged BEFORE Mask Application", edged)
        cv2.imshow("Edged AFTER Mask Application", edged_masked)
        #cv2.imshow("Harris Corner Detection", corners)
        
        key = cv2.waitKey(1)
        
        if key == 115: # "s" key
            break
        if key == 99: # "c" key
            now = datetime.now() # current data and time
            now = now.strftime("%m%d%Y_%H%M%S")
            
            img_name = "capture_" + now + ".png"
            img_name_frame = "capture_" + now + "_frame.png"
            
            cv2.imwrite(img_name, edged_masked)
            cv2.imwrite(img_name_frame, edged)
            print("img written: " + img_name)
    
cap.release()
cv2.destroyAllWindows()

