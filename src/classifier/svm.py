import cv2
import os
import numpy as np 

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import sys
sys.path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from extractor import hog
from extractor import orb
from extractor import sift
from extractor import surf
from tools import cross_validation

path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'
   
def split_sets(conj_train, conj_test, descriptor):

    for i in range(0, len(conj_test)):
        print("\n\n\n\n************************************")
        print("EXECUÇÃO "+ str(i+1) + " de ", end='')
        print(len(conj_test))
        print("************************************")
        
        if descriptor == 'hog':
            X_train, y_train = hog.extract(path, conj_train[i]);    
            X_test, y_test   = hog.extract(path, conj_test[i]);

        elif descriptor == 'orb':
            X_train, y_train = orb.extract(path, conj_train[i]);    
            X_test, y_test   = orb.extract(path, conj_test[i]);

        elif descriptor == 'sift':
            X_train, y_train = sift.extract(path, conj_train[i]);    
            X_test, y_test   = sift.extract(path, conj_test[i]);

        elif descriptor == 'surf':
            X_train, y_train = surf.extract(path, conj_train[i]);    
            X_test, y_test   = surf.extract(path, conj_test[i]);

        elif descriptor == 'combined':
            X_train_SIFT, y_train_SIFT  = sift.extract(path, conj_train[i]);
            X_test_SIFT, y_test_SIFT    = sift.extract(path, conj_test[i]);
            
            X_train_HOG, y_train_HOG    = hog.extract(path, conj_train[i]);
            X_test_HOG, y_test_HOG      = hog.extract(path, conj_test[i], train=False);
          
            X_train_ORB, y_train_ORB    = orb.extract(path, conj_train[i]);
            X_test_ORB, y_test_ORB      = orb.extract(path, conj_test[i], train=False);
            
            X_train_SURF, y_train_SURF  = surf.extract(path, conj_train[i]);
            X_test_SURF, y_test_SURF    = surf.extract(path, conj_test[i], train=False);
            
            
            X_train = np.concatenate( [ X_train_ORB, X_train_HOG, X_train_SIFT, X_train_SURF ], axis=1 );
            y_train = y_train_ORB;
            X_test  = np.concatenate( [ X_test_ORB, X_test_HOG, X_test_SIFT, X_test_SURF ], axis=1 );
            y_test  = y_test_ORB;


        init(X_train, y_train, X_test, y_test, i+1)
        
    

def classify(descriptor):
    conj_train, conj_test = cross_validation.split(10)
    split_sets(conj_train, conj_test, descriptor)
        

    

    
def init(X_train, y_train, X_test, y_test, index = 0):
    
    # pca = PCA(n_components=3, whiten=True)
    # pca = pca.fit(X_train)

    # print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix

    oc_svm_clf = svm.OneClassSVM(nu=0.1, gamma=0.0001)  # Obtained using grid search
    oc_svm_clf.fit(X_train, y_train)
    oc_svm_preds = oc_svm_clf.predict(X_test)
    

    cm = confusion_matrix(y_target=y_test, y_predicted=oc_svm_preds, binary=True)
    fig, ax = plot_confusion_matrix(conf_mat=cm)

    print(cm)

    # plt.savefig("confusion_matrix.pdf", format='pdf')
    plt.savefig("confusion_matrix"+str(index)+".png", format='png')

    ###############################################
    ## Classification Report
    ###############################################

    from sklearn.metrics import classification_report


    c_report = classification_report(y_test, oc_svm_preds)

    ### print values
    print("classification_report")
    print(c_report)

    # import pandas as pd 

    # c_report = classification_report(y_test, y_predicted, output_dict=True)
    # df_report = pd.DataFrame(c_report)
    # df_report.to_csv('.\\output\\report.csv', index= True)