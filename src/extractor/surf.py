import cv2
import os
import numpy as np 
from tqdm import tqdm


def extract(path, data, train=True):

        vector_size = 17256
        x_val = []
        y_val = []

        if train:
            print("SURF descripting train")
        else:
            print("SURF descripting test")


        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)
        
        for f in tqdm(data):        
            try:
                # Carrega a imagem pra memória
                img = cv2.imread(os.path.join(path, f), cv2.COLOR_RGB2GRAY)

                kps, dsc = surf.detectAndCompute(img, None)

                

                if dsc is not None:
                    dsc = dsc.flatten()
                    needed_size = (vector_size * 64)

                    if dsc.size < needed_size:
                        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
                    
                    x_val.append(dsc)
                    
                    # get class
                    y_val.append(int(f[:1]))
             

            except cv2.error as e:
                print('Error: ', e)
                
        
        print(len(x_val[0]))
        
                    
        return np.array(x_val), np.array(y_val)