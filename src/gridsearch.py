from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
sys.path
from extractor import sift
from sklearn.svm import SVC


path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'
X_train, y_train, X_test, y_test = sift.extract(path);


grid_params = {
    'n_neighbors': [11, 5, 7, 3, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'] 
}

gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    verbose=1,
    cv=3,
    n_jobs=-1
)

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores=['recall', 'precision']

# for score in scores:
#     gs = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                     scoring='%s_macro' % score, n_jobs=-1)

#     gs_results = gs.fit(X_train, y_train)

#     print(gs_results.best_score_)
#     print(gs_results.best_estimator_)
#     print(gs_results.best_params_)

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)

gs_results = gs.fit(X_train, y_train)

print(gs_results.best_score_)
print(gs_results.best_estimator_)
print(gs_results.best_params_)