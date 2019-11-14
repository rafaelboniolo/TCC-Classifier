# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# # img = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0080.png", 0)
# # img1 = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0080.png",0)          # queryImage
# # img2 = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0085.png",0) # trainImage
# img = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_007_0043.png", cv2.COLOR_RGB2GRAY)


# # Initiate SIFT detector
# orb = cv2.ORB_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img,None)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors.
# matches = bf.match(des1)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.

# img3 = cv2.drawMatches(img,kp1,matches[:200], None, flags=2)

# plt.imshow(img3),plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_001_0083.png',0)

# Initiate STAR detector
orb = cv2.ORB_create(3000)

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,outImage=None, color=(0,255,0), flags=0)
plt.imshow(img2, cmap=plt.cm.gray),plt.show()