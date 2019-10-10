import cv2
import os
import numpy as np 

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from extractor import hog
from extractor import orb
from extractor import sift
from extractor import surf

def classify(type):

    path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'
    
    if type == "orb":
        X_train, y_train = orb.extract(path);
        X_test,  y_test  = orb.extract(path);
    
    elif type == "hog":
        X_train, y_train, X_test, y_test = hog.extract(0.7, path, '.png');    
    
    elif type == "sift":
        X_train, y_train = sift.extract(path);
    
    elif type == "surf":
        X_train, y_train, X_test, y_test = surf.extract(path);
    
        
    

    # pca = PCA(n_components=2, whiten=True)
    # pca = pca.fit(X_train)

    # print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=7)

    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix

    cm = confusion_matrix(y_target=y_test, y_predicted=y_predicted, binary=True)
    fig, ax = plot_confusion_matrix(conf_mat=cm)

    print(cm)

    plt.savefig("confusion_matrix.pdf", format='pdf')
    plt.savefig("confusion_matrix.png", format='png')

    ###############################################
    ## Classification Report
    ###############################################

    from sklearn.metrics import classification_report


    c_report = classification_report(y_test, y_predicted)

    ### print values
    print("classification_report")
    print(c_report)

    import pandas as pd 

    c_report = classification_report(y_test, y_predicted, output_dict=True)
    df_report = pd.DataFrame(c_report)
    df_report.to_csv('report.csv', index= True)