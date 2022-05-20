'''
Senior Design, Group MEM 23 Initial Testing Scripts
Group: L. Heisler, J. Bunch, C. Silvia, R. Zegarski, J. Heinzman
'''

__author__ = "Liam Heisler, MEM 23"
__version__ = "x.x"
__status__ = "Development"

# Computer vision imports
import cv2
import cv2.aruco as aruco

# Utility Imports
import numpy as np, imutils, os, sys, math, traceback
#from imutils.video import VideoStream
from datetime import datetime
from imutils import perspective
from imutils import contours

# Grab Aruco dictionary (pattern recognition) & Standard Params
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters_create() # replace with ours later
marker_size = 50 #mm

# Get camera parameters (intrinsic matrix, dist coeff)
# looks for two files in /calibration/
usb_mtx_file = os.getcwd() + "\calibration" + "\\usb_mtx.npy"
usb_dist_file = os.getcwd() + "\calibration" + "\\usb_dist.npy"

usb_mtx = np.load(usb_mtx_file)
usb_dist = np.load(usb_dist_file)

# Misc params (clean later!)
#lbls = ["X: ", "Y: ", "Z: ", "Roll (deg.): "]
lbls = ["ID: ", "X: ", "Y: ", "Z: ", "Roll (deg.): ", "Pitch (deg.): ", "Yaw (deg.): "]

    
def findCameras():
    # check first 3 indexes
    print("\n ---------- CAMERA SELECTION ---------- \n")
    index = 0
    arr = []
    i = 3
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]: # ret check
            arr.append(index)
            cap.release()
        index += 1
        i -= 1

    if len(arr) == 0:
        print("[INFO] > No camera detected!")
        print("\n -------------------------------------- \n")
        return None
    elif len(arr) == 1:
        print("[INFO] > Camera detected! Using cam at idx: {}".format(arr[0]))
        print("\n -------------------------------------- \n")
        return arr[0]
    elif len(arr) > 1:
        print("[INFO] > Available indexes: {} \n".format([x for x in arr]))
        selection = int(input("[USR]  > Enter desired camera idx: "))
        print("\n -------------------------------------- \n")
        return selection

def marker_selector():
    # Select an Aruco marker to identify in the image
    text_based = True # use text based selector for now!
    img_based = True

    avail_mkrs = [0, 1, 2] # handles up to 6 different Aruco mkrs

    # To be replaced with GUI or on the CV2 frame!    
    print("\n ---------- MARKER SELECTION ---------- \n")
    print("[INFO] > Available indexes: {} \n".format([x for x in avail_mkrs]))
    selection = int(input("[USR]  > Enter desired marker no.: "))
    print("\n -------------------------------------- \n")
    return selection

# Camera configuration
cap = cv2.VideoCapture(findCameras())
camW = 1920 # camera pixel width
camH = 1080 # camera pixel height
#camRate = 40 # camera's frame rate

# Set camera config parameters
use = False
if use:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camH)
    #cap.set(5, camRate)

def findOffsets(rvec, tvec):
    x = round(0.1 * tvec[0][0][0], 3) #cm
    y = round(0.1 * tvec[0][0][1], 3) #cm
    z = round(0.1 * tvec[0][0][2], 3) #cm
    
    return [x, y, z]

def findAngles(rvec, tvec):
    if rvec is not None:
        rmat = cv2.Rodrigues(rvec)[0]

        # hstack the rot. matrix and t vec (extrinsic/projection matrix)
        P = np.hstack((rmat, tvec[0].T))
        #P = np.concatenate((rmat,tvec[0].T),axis=1)

        # Decompose the projection matrix into our Euler Angles        
        euler_angles_deg = -cv2.decomposeProjectionMatrix(P)[6] # 6th item in output are the Euler Angles

        # I believe it's X (roll), Y (pitch), Z (yaw) -> VERIFY THIS!
        roll = round(euler_angles_deg[2][0], 2)
        pitch = round(euler_angles_deg[0][0], 2)
        yaw = round(euler_angles_deg[1][0],2)
        
        return [roll, pitch, yaw]
    else:
        return "ERR", "ERR", "ERR"
    pass


# What marker do we want to detect in the image?
desired_mkr_id = marker_selector()

while True:
    # Capture a camera frame and whether or not it's been captured (ret)
    ret, frame = cap.read()
    h, w, c = frame.shape # grab the frames size in h,w

    if ret: # do we have a valid feed/img?
        # Convert camera frame to grayscale frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the aruco markers in the grayscale img (use our params?)      
        corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict,
                                                     parameters=aruco_params,
                                                     cameraMatrix=usb_mtx,
                                                     distCoeff=usb_dist)
        
        if ids is not None: # we have found some markers in the img
            # Calculate pose of the camera wrt to the object and display the axis on the marker
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, usb_mtx, usb_dist)    

            # Temporary bug fix to remove bad rvecs and tvecs 
            if rvec.shape[0] < 2 or tvec.shape[0] < 2:
                try:
                    cv2.aruco.drawAxis(frame, usb_mtx, usb_dist, rvec, tvec, marker_size)
                except Exception:
                    print(traceback.format_exc())

                # Gather the translational offsets
                trans_offsets = [ids[0][0]] + findOffsets(rvec, tvec)

                # Gather the angular offsets (Euler angles)
                angle_offsets = findAngles(rvec, tvec)

                # Combine both offsets to one array
                all_offsets = trans_offsets + angle_offsets

                # Display all offsets on the frame
                frame_y_pos = 0
                for i in range(len(lbls)):
                    mkr_offset = lbls[i] + str(all_offsets[i])
                    frame_y_pos += int(h / len(lbls)) - 25
                    cv2.putText(frame, mkr_offset, (20, frame_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("[INFO]: Bad rvec/tvec detected")
	    
            
    cv2.imshow("Aruco Detection", frame)

    key = cv2.waitKey(1)
    # print(key) # Useful for figuring out key IDs
    if key == 115: # "s" key
        break
    if key == 99: # "c" key
        now = datetime.now() # current data and time
        now = now.strftime("%m%d%Y_%H%M%S")
        img_name = "arcuoCap_" + now + ".png"
        cv2.imwrite(img_name, frame)
        print("img written: " + img_name)
    if key == 109: # "m" key
        desired_mkr_id = marker_selector()

cap.release()
cv2.destroyAllWindows()

            
