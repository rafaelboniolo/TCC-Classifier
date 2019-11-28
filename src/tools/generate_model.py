import sys
sys.path

from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from tools import cross_validation
from extractor import hog
from extractor import orb
from extractor import sift
from extractor import surf


path = 'C:\\Users\\rafae\\Documents\\GitHub\\TCC-Dataset\\dataset'
   
def split_sets(conj_train, conj_test, classifier):

    for i in range(0, len(conj_test)):
        print("\n\n\n\n************************************")
        print("EXECUÇÃO "+ str(i+1) + " de ", end='')
        print(len(conj_test))
        print("************************************")
        
        X_train, y_train = hog.extract(path, conj_train[i]);    
        X_test, y_test   = hog.extract(path, conj_test[i])

        init(X_train, y_train, X_test, y_test, i+1, classifier)
    
        
    

def save():
    classifier = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="manhattan", n_jobs=-1)
    conj_train, conj_test = cross_validation.split(10)
    split_sets(conj_train, conj_test, classifier)    

    
def init(X_train, y_train, X_test, y_test, index = 0, classifier=None):
    classifier.fit(X_train, y_train)

    if index == 9:
        dump(classifier, 'model.joblib')

   
