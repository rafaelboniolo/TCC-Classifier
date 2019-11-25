from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
sys.path
from extractor import orb
from extractor import sift
from extractor import hog
from extractor import surf
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import numpy as np 


path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'

data = np.array([f for f in os.listdir(path) if(f.endswith('.png'))])

X_train, y_train  = sift.extract(path, data);
# X_train_hog, y_train_hog    = hog.extract(path, data);
# X_train_sift, y_train_sift  = sift.extract(path, data);
# X_train_orb, y_train_orb    = orb.extract(path, data);

# X_train = np.concatenate( [ X_train_orb, X_train_hog, X_train_sift, X_train_surf ], axis=1 );
# y_train = y_train_hog

# grid_params = {
#     'n_neighbors': [11, 5, 7, 3, 19, 13, 15],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan'] 
# }

# pca = PCA(n_components=3, whiten=True)
# pca = pca.fit(X_train)

# print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
# X_train = pca.transform(X_train)

# gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=10, n_jobs=-1)

# gs_results = gs.fit(X_train, y_train)

# print(gs_results.best_score_)
# print(gs_results.best_estimator_)
# print(gs_results.best_params_)


tuned_parameters = {'gamma' : [0.0001, 0.01, 0.1],
        'nu' : [0.1, 0.5, 0.9],
        'kernel':['rbf', 'linear']}
        # 'rbf', 'linear', 'sigmoid', 'poly'

scores=['precision']

for score in scores:
    gs = GridSearchCV(OneClassSVM(), tuned_parameters, cv=10,
                    scoring='%s_macro' % score, n_jobs=-1)

    gs_results = gs.fit(X_train, y_train)

    print(gs_results.best_score_)
    print(gs_results.best_estimator_)
    print(gs_results.best_params_)

