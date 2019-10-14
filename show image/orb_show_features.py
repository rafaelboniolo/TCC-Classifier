import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0080.png", 0)
# img1 = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0080.png",0)          # queryImage
# img2 = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_003_0085.png",0) # trainImage
img1 = cv2.imread("C:\\img\\0.png",0) # trainImage
img2 = cv2.imread("C:\\img\\00.png",0) # trainImage


# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print("as")

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:200], None, flags=2)

plt.imshow(img3),plt.show()

print("as")


