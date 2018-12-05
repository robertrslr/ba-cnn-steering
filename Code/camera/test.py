import sys
sys.path.append("../")

from ueye_cam import ueye_cam
import cv2

cam = ueye_cam()

while(True):
    cv2.imshow("ad", cam.read())
    cv2.waitKey(1)
