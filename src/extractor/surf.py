import cv2
import os
import numpy as np 
from tqdm import tqdm


def extract(path, data, train=True):
        
        vector_size = 32  
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
                img = cv2.resize(img, (256, 256))

                # kps, dsc = surf.detectAndCompute(img, None)
                
                kps = surf.detect(img)

                kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

                kps, dsc = surf.compute(img, kps)
                

                if dsc is not None:
                    dsc = dsc.flatten()
                    
                    needed_size = (vector_size * 64)
                    if dsc.size < needed_size:
                    #     # if we have less the 32 descriptors then just adding zeros at the
                    #     # end of our feature vector
                        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
                    
                    x_val.append(dsc)
                    
                    # get class
                    if int(f[:1]) == 0:
                        y_val.append(-1)
                    else:
                        y_val.append(1)
             

            except cv2.error as e:
                print('Error: ', e)
                          
        return np.array(x_val), np.array(y_val)