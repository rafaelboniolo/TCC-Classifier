import sys
from classifier import knn
from classifier import svm

extractor = sys.argv[1]
algoritm = sys.argv[2]

if  not (extractor  ==  "orb" or extractor == 'hog'):
    print("invalid extractor argument!")
    exit()

if algoritm == 'knn':
    print('knn', extractor)
    knn.classify(extractor)

elif algoritm == 'svm':
    print('svm', extractor)
    svm.classify(extractor)
else:
    print("invalid classifier argument!")
    exit()
