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

cap = cv2.VideoCapture(1)

while True:
    _, frame, = cap.read()
    
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply edge detection
    edged = cv2.Canny(gray, 50, 100)

    # Neccessary? Dilated and eroded to ensure gaps are closed
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the image (enclosed areas)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours) 

    # loop thru contours to gather ones that are enclosed shapes
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        #print(len(approx))
        if len(approx) == 6: # 5 corners on mfdo
            cv2.drawContours(frame, [cnt], 0, 255, -1)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Edged", edged)
    
    key = cv2.waitKey(1)
    
    if key == 115: # "s" key
        break
    if key == 99: # "c" key
        now = datetime.now() # current data and time
        now = now.strftime("%m%d%Y_%H%M%S")
        img_name = "capture_" + now + ".png"
        cv2.imwrite(img_name, frame)
        print("img written: " + img_name)
    
cap.release()
cv2.destroyAllWindows()
