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
import imutils, time
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist # used in imutils

import numpy as np, math
from time import sleep

def define_cameras(usb=False, piCam=False):
    
    # define the USB camera
    if usb:
        cam = cv2.VideoCapture(0)
    if piCam:
        cam = PiCamera()
        cam.resolution = (2592, 1944)
        cam.framerate = 15
    return cam

def img_capture():
    
    
    usbCam = True
    piCam = False

    cam = define_cameras(usbCam,piCam)
    
    if usbCam:
        ret, frame = cam.read()
        if not ret:
            print("Image failed to read!")

        cv2.imshow("Single Image Capture - Captured on exec!", frame)
            
        img_name = "opencv_frame_singleCap.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        cam.release()
        cv2.destroyAllWindows()

        return frame
    else:
        frame = pi_cam_capture(cam)
        return frame

def multi_img_capture():

    cam = define_cameras(True,False)

    cv2.namedWindow("Multiple Image Captures - SPACE to capture!")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    
def pi_cam_capture(cam):
    
    # initialize the camera and grab a reference to the raw camera capture
    rawCapture = PiRGBArray(cam)
    # allow the camera to warmup
    time.sleep(0.5)
    # grab an image from the camera
    cam.capture(rawCapture, format="bgr")
    frame = rawCapture.array
    # display the image on screen and wait for a keypress
    #cv2.imshow("Image", frame)
    #cv2.waitKey(0)
    
    return frame

def find_mfdo(img):
    # Grab a test image
    #img = single_img_capture()

    # Set it to gray and smooth it via Gaussian (reduce noise)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform Canny edge detection
    edged = cv2.Canny(gray, 50, 100)

    # Neccessary? Dilated and eroded to ensure gaps are closed
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cv2.imshow("test",edged)
    cv2.imwrite("edged_img.png", edged)

    # Find the contours the map of edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def calc_distance(mfdo_marker):
    # known distance
    focal_length = 3.6 #mm
    #focal_length = 2571.43 #3.6mm focal length conv to px via 1.4um/px
    #focal_length = 714.29 #1mm focal length (lenovo laptop) conv to px via 1.4um/px (estimate!!)
    real_height = 44.45 #mm (4.75in square)
    img_height = 1944 #px, resolution set to 2592x1944px
    obj_height = mfdo_marker[1][1]
    sensor_height = 2.74 #mm, 3.76x2.74mm
    
    #return (known_width*focal_length)/mfdo_marker[1][1]
    return (focal_length*real_height*img_height)/(obj_height*sensor_height)

# Grab a test image
image = img_capture()

# Identify the MFDO in the image (WIP*)
mfdo_marker = find_mfdo(image)

# Calculate distance to object
dist = calc_distance(mfdo_marker)
print("Marker height(px): " + str(mfdo_marker[1][1]))
print("Dist to obj (in): " + str(dist*0.0393701))

marker = mfdo_marker
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fin" % (dist*0.0393701),
            (image.shape[1] - 300, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 3)
cv2.imshow("image", image)
#cv2.waitKey(0)


# Set it to gray and smooth it via Gaussian (reduce noise)
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##gray = cv2.GaussianBlur(gray, (7, 7), 0)
##
### Perform Canny edge detection
##edged = cv2.Canny(gray, 50, 100)
##
### Neccessary? Dilated and eroded to ensure gaps are closed
##edged = cv2.dilate(edged, None, iterations=1)
##edged = cv2.erode(edged, None, iterations=1)
##
### Find the contours the map of edges
##cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
##	cv2.CHAIN_APPROX_SIMPLE)
##cnts = imutils.grab_contours(cnts)
## 
##
##c = max(cnts, key=cv2.contourArea)
