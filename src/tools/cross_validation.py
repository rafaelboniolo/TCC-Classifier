import numpy as np
from sklearn.model_selection import KFold
import os
import cv2
import sys
sys.path
from tqdm import tqdm


path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'

def split(folds = 2):
    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(".png"))])
    imgs = []

    print("Splitting dataset " + str(folds) + " folds...")

    for f in tqdm(f_imgs):
        try:

            img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))

            imgs.append(img)

        except KeyboardInterrupt:
            exit()
        except:
            pass

    kf = KFold(folds, shuffle=True, random_state=0)

    conj_train = []
    conj_test = []

    for train, test in kf.split(imgs):
        conj_train.append(f_imgs[train])
        conj_test.append(f_imgs[test])

    return conj_train, conj_test
