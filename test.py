import cv2
import os
import numpy as np 

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

## data path for load

train_path = "C:\\Users\\rafae\\Desktop\\TCC\\WebScrapping\\src\\carro"
test_path = "Food-5K/evaluation"
val_path = "Food-5K/validation"


f_imgs = np.array([f for f in os.listdir(train_path) if(f.endswith('.png'))])

img = cv2.imread(os.path.join(train_path, f_imgs[0]), cv2.COLOR_RGB2GRAY)

cv2.imshow ("img", img) 
cv2 . waitKey ( 0 ) 