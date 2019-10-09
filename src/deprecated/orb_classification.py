## https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774

## https://ianlondon.github.io/blog/how-to-sift-opencv/
## https://scikit-learn.org/0.15/auto_examples/svm/plot_oneclass.html
## https://mccormickml.com/2013/05/09/hog-person-detector-tutorial/

## https://gilscvblog.com/2013/08/18/a-short-introduction-to-descriptors/

## https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

import cv2
import os
import numpy as np 

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

## data path for load

train_path = "C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset"

## dataset TCC

from descriptor import extract_features_hog

X_train, y_train, X_test, y_test = extract_features_hog(0.7, train_path, '.png')
# X_train, y_train, X_test, y_test = extract_features_keypoints('ORB', 0.8, train_path, '.png')

## classification

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# Apply standard scaler to output from resnet50
# ss = StandardScaler()
# ss.fit(X_train)
# X_train = ss.transform(X_train)
# X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=2, whiten=True)
pca = pca.fit(X_train)

# print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# plot vector features
from plot import visualize_pca

visualize_pca(X_train, y_train)

# Train classifier and obtain predictions for OC-SVM
# classifier = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
# classifier = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

#### f1-score
## -1 0.00

## IsolationForest
## -1 0.20   L2-Hys
## -1 0.36   L2
## -1 0.14   L1
## -1 0.22   L1-sqrt

## KNN
## -1 0.46 K=3
## -1 0.47 k=5
## -1 0.48 k=7


classifier.fit(X_train, y_train)
#if_clf.fit(X_train)

y_predicted = classifier.predict(X_test)
#if_preds = if_clf.predict(X_test)

#print(y_test)
#print(oc_svm_preds)

###############################################
## Confusion Matrix
## http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
###############################################

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix



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