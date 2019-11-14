import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

def visualize_pca3D(X_train, y_train):
    
    ## import
    from mpl_toolkits.mplot3d import Axes3D

    # Visualize
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    classes = np.sort(np.unique(y_train))
    labels = ["Anomalia", "Normal"]
    colors = [(0.0, 0.63, 0.69), 'black']
    markers = ["o", "D"]

    for class_ix, marker, color, label in zip(classes, markers, colors, labels):
        ax.scatter(
                    X_train[np.where(y_train == class_ix), 0],
                    X_train[np.where(y_train == class_ix), 1],
                    X_train[np.where(y_train == class_ix), 2],
                    marker=marker, color=color, label=label)
        ax.legend(loc='best')

    ax.set_xlabel('Componente Principal 1', fontsize = 15)
    ax.set_ylabel('Componente Principal 2', fontsize = 15)
    ax.set_zlabel('Componente Principal 3', fontsize = 15)

    plt.title('3 primeiros componentes principais', fontsize = 20)

    plt.savefig("pca3D.pdf", format='pdf')
    plt.savefig("pca3D.png", format='png')
