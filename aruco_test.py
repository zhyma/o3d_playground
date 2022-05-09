import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

## make sure opencv-python is not installed
## install opencv-contrib-python

# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
# fig = plt.figure()
# nx = 10
# ny = 9
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")

# # plt.savefig("_data/markers.pdf")
# plt.show()


frame = cv2.imread("test2/image.jpeg")
# plt.figure()
# plt.imshow(frame)
# plt.show()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

print(corners)
print(ids)
print(frame_markers)