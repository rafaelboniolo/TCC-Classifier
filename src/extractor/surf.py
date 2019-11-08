import cv2
import os
import numpy as np 
from tqdm import tqdm


def extract(path, data, train=True):
        vector_size = 20000
        x_val = []
        y_val = []

        if train:
            print("SURF descripting train")
        else:
            print("SURF descripting test")


        surf = cv2.xfeatures2d.SURF_create()
        
        for f in tqdm(data):        
            try:
                # Carrega a imagem pra memória
                img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)

                # kps, dsc = surf.detectAndCompute(img, None)
                
                kps = surf.detect(img)

                kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

                kps, dsc = surf.compute(img, kps)
                

                if dsc is not None:
                    dsc = dsc.flatten()

                    if dsc.size < vector_size:
                        dsc = np.concatenate([dsc, np.zeros(vector_size - dsc.size)])
                    else:
                        dsc = dsc[:vector_size]
                    
                    x_val.append(dsc)
                    
                    # get class
                    y_val.append(int(f[:1]))
             

            except cv2.error as e:
                print('Error: ', e)
                          
        return np.array(x_val), np.array(y_val)