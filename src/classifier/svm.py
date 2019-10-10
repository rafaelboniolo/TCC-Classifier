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

def classify(type):
    path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'
    
    if type == "orb":
        X_train, y_train = orb.extract(path);
        X_test,  y_test  = orb.extract(path);
    else:
        X_train, y_train, X_test, y_test = hog.extract(0.7, path, '.png');
        
    pca = PCA(n_components=2, whiten=True)
    pca = pca.fit(X_train)

    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix

    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
    oc_svm_clf.fit(X_train, y_train)
    oc_svm_preds = oc_svm_clf.predict(X_test)
    print(y_test)
    print(oc_svm_preds)


    cm = confusion_matrix(y_target=y_test, y_predicted=oc_svm_preds, binary=True)
    fig, ax = plot_confusion_matrix(conf_mat=cm)

    plt.savefig("oc_svm_confusion_matrix.png")

