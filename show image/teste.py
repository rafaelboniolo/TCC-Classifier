import cv2
import os
import numpy as np 
from sklearn.utils import shuffle
from skimage.feature import hog
import sys
sys.path
from tqdm import tqdm
from sklearn.decomposition import PCA



def extract():

    path = "C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset"

    x_val = []
    y_val = []

    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(".png"))])
     
    for f in tqdm(f_imgs):
        
        try:
            img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))

            if img is not None:

                # features = hog(img, orientations=8, pixels_per_cell=(img.shape[0]/8,img.shape[1]/8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
                features = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)

                # print(features.size, features)
                x_val.append(features)
                
                # get class
                if int(f[:1]) > 0: 
                    y_val.append(1)
                else:
                    y_val.append(-1)
        
        except cv2.error as e:
            print(f)
            print('Error: ', e)
                
    
    return np.array(x_val), np.array(y_val)

X_train, y_train = extract()

pca = PCA(n_components=3, whiten=True)
pca = pca.fit(X_train)
print("Treinando PCA...")

from plot import visualize_pca3D

visualize_pca3D(X_train, y_train)