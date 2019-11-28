import sys
from classifier import knn
from classifier import svm
from tools import generate_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algoritm", required=False, help="Entre com o algoritmo de classificação")
ap.add_argument("-d", "--descriptor", required=False, help="Entre com o descritor")
ap.add_argument("-s", "--save", required=False, help="Salvar o modelo de predição")

args = vars(ap.parse_args())


descriptor = args["descriptor"]
algoritm  =  args["algoritm"]
save  =  args["save"]

if save : 
    generate_model.save()


if  not (descriptor  ==  "orb" or descriptor == 'hog' or descriptor == 'sift' or descriptor == 'surf' or descriptor == 'combined'):
    print("invalid extractor argument!")
    exit()

if algoritm == 'knn':
    print('knn', descriptor)
    knn.classify(descriptor)

elif algoritm == 'svm':
    print('svm', descriptor)
    svm.classify(descriptor)
    
else:
    print("invalid classifier argument!")
    exit()
