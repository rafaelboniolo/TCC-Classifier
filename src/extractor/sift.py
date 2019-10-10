import cv2
import os
import numpy as np 

def extract(path, vector_size = 32):
        
        x_val = []
        y_val = []

        sift = cv2.xfeatures2d.SIFT_create()

        print(path)
        f_imgs = np.array([f for f in os.listdir(path) if(f.endswith('.png'))])
        
        i = 0
        for f in f_imgs:
        
            try:
                # Carrega a imagem pra memória
                img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)

                kps, dsc = sift.detectAndCompute(img, None)

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
                    y_val.append(int(f[:1]))
                i = i + 1
                print(str(i));

            except cv2.error as e:
                print('Error: ', e)
                    
        return x_val, y_val