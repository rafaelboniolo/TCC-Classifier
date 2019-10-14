import numpy as np
from sklearn.model_selection import KFold
import os
import cv2

path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'

def split(folds = 2):
    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith("010.png"))])
    imgs = []

    for f in f_imgs:
        try:

            print(os.path.join(path, f))

            img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))

            imgs.append(img)

        except:
            pass

    kf = KFold(folds)


    return kf.split(imgs)
