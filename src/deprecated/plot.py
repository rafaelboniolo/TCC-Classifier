import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

def visualize_pca(X_train, y_train):

    # Visualize
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)

    classes = np.sort(np.unique(y_train))
    labels = ["Anomaly", "Normal"]
    colors = [(0.0, 0.63, 0.69), 'black']
    markers = ["o", "D"]

    for class_ix, marker, color, label in zip(classes, markers, colors, labels):
        ax.scatter(X_train[np.where(y_train == class_ix), 0],
                   X_train[np.where(y_train == class_ix), 1],
                   marker=marker, color=color, edgecolor='whitesmoke',
                   linewidth='1', alpha=0.9, label=label)
        ax.legend(loc='best')

    plt.xlabel('Principal Component 1', fontsize = 15)
    plt.ylabel('Principal Component 2', fontsize = 15)
    plt.title('2 first principal components', fontsize = 20)

    plt.savefig("pca.pdf", format='pdf')
    plt.savefig("pca.png", format='png')
