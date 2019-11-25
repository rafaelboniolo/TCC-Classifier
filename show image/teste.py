import cv2
import os
import numpy as np 
from sklearn.utils import shuffle
from skimage.feature import hog
from skimage.feature import hog
from skimage.feature import hog
from skimage.feature import hog
import sys
sys.path
from tqdm import tqdm
from sklearn.decomposition import PCA



def extract():

    path = "C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset"

    x_val = []
    y_val = []
    vector_size= 32
    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(".png"))])
     
    sift = cv2.xfeatures2d.SIFT_create(11)

    for f in tqdm(f_imgs):
    
        try:
            # Carrega a imagem pra memória
            img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)

            kps, dsc = sift.detectAndCompute(img, None)

            if dsc is not None:

                # # Flatten all of them in one big vector - our feature vector
                dsc = dsc.flatten()
                # # Making descriptor of same size
                # # Descriptor vector size is 64
                needed_size = (vector_size * 64)
                if dsc.size < needed_size:
                #     # if we have less the 32 descriptors then just adding zeros at the
                #     # end of our feature vector
                    dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

                # # print(dsc.size)
                x_val.append(dsc)
                
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