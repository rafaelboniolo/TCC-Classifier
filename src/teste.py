import numpy as np
from sklearn.model_selection import KFold
import os
import cv2
import sys
sys.path
from tqdm import tqdm

from extractor import hog
from extractor import orb
from extractor import sift
from extractor import surf


path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'


f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(".png"))])

X_train_SIFT, y_train_SIFT  = sift.extract(path, f_imgs);
print(np.array(X_train_SIFT).shape)

X_train_HOG, y_train_HOG    = hog.extract(path, f_imgs);
print(np.array(X_train_HOG).shape)

X_train_ORB, y_train_ORB    = orb.extract(path, f_imgs);
print(np.array(X_train_ORB).shape)

X_train_SURF, y_train_SURF  = surf.extract(path, f_imgs);
print(np.array(X_train_SURF).shape)


X_train = np.concatenate( [ X_train_ORB, X_train_HOG, X_train_SIFT, X_train_SURF ], axis=1 );

print(X_train.shape)

# hog.extract(path, f_imgs);
# orb.extract(path, f_imgs);
# surf.extract(path, f_imgs);