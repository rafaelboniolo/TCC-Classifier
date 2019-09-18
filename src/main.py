
import cv2
import os
import numpy as np 

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import sys
sys.path

from ExtractFeatures import ExtractFeatures
from Mode import Mode

X_train, y_train = ExtractFeatures.byOrb(Mode.TRAIN.value);
X_test,  y_test  = ExtractFeatures.byOrb(Mode.TRAIN.value);



## classification

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm

# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
# pca = PCA(n_components=512, whiten=True)
pca = PCA(whiten=True)

pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Train classifier and obtain predictions for OC-SVM
oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
#if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

oc_svm_clf.fit(X_train, y_train)
#if_clf.fit(X_train)

oc_svm_preds = oc_svm_clf.predict(X_test)
#if_preds = if_clf.predict(X_test)

print(y_test)
print(oc_svm_preds)

###############################################
## Confusion Matrix
## http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
###############################################

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_target=y_test, y_predicted=oc_svm_preds, binary=True)
fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.savefig("oc_svm_confusion_matrix.png")