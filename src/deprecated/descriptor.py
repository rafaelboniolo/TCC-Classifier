## https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

import cv2
import os
import numpy as np 

from sklearn.utils import shuffle

## https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
## https://mccormickml.com/2013/05/09/hog-person-detector-tutorial/
## https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py

## Parallel
## https://scikit-image.org/docs/dev/user_guide/tutorial_parallelization.html
## 

from skimage.feature import hog

def extract_features_hog(train_rate, path, posfix):

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


## feature extraction orb

def extract_features_keypoints(alg, train_rate, path, posfix, vector_size = 32):

    x_val = []
    y_val = []

    if alg == 'ORB':
        alg = cv2.ORB_create()
    else:
        alg = cv2.KAZE_create()

    # list images name
    f_imgs = np.array([f for f in os.listdir(path) if(f.endswith(posfix))])
        
    for f in f_imgs:
    
        try:

            print(os.path.join(path, f))

            img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (128, 128))

            # Dinding image keypoints
            kps = alg.detect(img)

            # Getting first 32 of them. 
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)   
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

            # computing descriptors vector
            kps, dsc = alg.compute(img, kps)

            if dsc is not None:

                # Flatten all of them in one big vector - our feature vector
                dsc = dsc.flatten()
                # Making descriptor of same size
                # Descriptor vector size is 64
                needed_size = (vector_size * 64)
                if dsc.size < needed_size:
                    # if we have less the 32 descriptors then just adding zeros at the
                    # end of our feature vector
                    dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

                # print(dsc.size)
                x_val.append(dsc)
                
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