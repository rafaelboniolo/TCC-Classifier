import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt


surf = cv2.xfeatures2d.SURF_create(400)

img = cv2.imread("C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset\\1_007_0043.png", cv2.COLOR_RGB2GRAY)

kps, dsc = surf.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, kps, None);

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)


plt.show()

# cv2.imshow("Image", img, cmap=plt.cm.gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
