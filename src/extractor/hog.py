import cv2
import os
import numpy as np 
from sklearn.utils import shuffle
from skimage.feature import hog

def extract(train_rate, path, posfix):

    x_val = []
    y_val = []

    # list images name
    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(posfix))])
        
    for f in f_imgs:
    
        try:

            print(os.path.join(path, f))

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
            print('Error: ', e)
                
    X, Y = np.array(x_val) , np.array(y_val)

    # split between training and testing data
    index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
    index_test  = list(set(range(X.shape[0])) - set(index_train))
                                
    X, Y = shuffle(X, Y)

    X_train, y_train = X[index_train], Y[index_train]
    X_test, y_test = X[index_test], Y[index_test]

    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test