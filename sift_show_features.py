import cv2
import os
import numpy as np 

sift = cv2.xfeatures2d.SIFT_create()

img = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_007_0043.png", cv2.COLOR_RGB2GRAY)

kps, dsc = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, kps, None);

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
