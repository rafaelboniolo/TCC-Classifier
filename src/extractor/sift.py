import cv2
import os
import numpy as np 
from sklearn.utils import shuffle


def extract(path, vector_size = 32):
        
        x_val = []
        y_val = []

        sift = cv2.xfeatures2d.SIFT_create()

        print(path)
        f_imgs = np.array([f for f in os.listdir(path) if(f.endswith('10.png'))])
        
        i = 0
        for f in f_imgs:
        
            try:
                # Carrega a imagem pra memória
                img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)

                kps, dsc = sift.detectAndCompute(img, None)

                if dsc is not None:

                    # # Flatten all of them in one big vector - our feature vector
                    # dsc = dsc.flatten()
                    # # Making descriptor of same size
                    # # Descriptor vector size is 64
                    # needed_size = (vector_size * 64)
                    # if dsc.size < needed_size:
                    #     # if we have less the 32 descriptors then just adding zeros at the
                    #     # end of our feature vector
                    #     dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

                    # # print(dsc.size)
                    x_val.append(dsc)
                    
                    # get class
                    y_val.append(int(f[:1]))
                i = i + 1
                print(str(i));

            except cv2.error as e:
                print('Error: ', e)
                    
        X, Y = np.array(x_val) , np.array(y_val)

        # split between training and testing data
        index_train = np.random.choice(X.shape[0], int(X.shape[0] * 0.7), replace=False)
        index_test  = list(set(range(X.shape[0])) - set(index_train))
                                    
        X, Y = shuffle(X, Y)

        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]

        print("Train:", X_train.shape, y_train.shape)
        print("Test: ", X_test.shape, y_test.shape)

        return X_train, y_train, X_test, y_test