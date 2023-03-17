import pathlib
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def get_min_size(dir_loc=None):
    if dir_loc is None:
        return False
    data_dir = pathlib.Path(dir_loc)
    data_dir = data_dir.glob('*.jpg')

    min_h = sys.maxsize
    min_l = sys.maxsize

    x = []
    y = []

    for path in data_dir:
        img = tf.keras.preprocessing.image.load_img(path)
        image_array = tf.keras.preprocessing.image.img_to_array(img)
        min_h = min(image_array.shape[0], min_h)
        min_l = min(image_array.shape[1], min_l)
        x.append(image_array.shape[0])
        y.append(image_array.shape[1])

    plt.plot(x, y, 'x')
    plt.show()


    return min_h, min_l


def consine_distances(X, Y):
    normalize_X = tf.nn.l2_normalize(X, 0)
    normalize_Y = tf.nn.l2_normalize(Y, 0)
    sim = tf.compat.v1.losses.cosine_distance(normalize_X, normalize_Y, dim=0)
    return sim

def plot_bar():
    k_acc = 77.2
    svm_acc = 77.8
    k_f1_score = 77.2
    svm_f1_score = 77.8
    ffn_acc = 80.7
    ffn_f1 = 80.7
    X = np.arange(3)
    fig, ax = plt.subplots()
    ax.bar(X + 0.00, [k_acc, svm_acc, ffn_acc], color='b', width=0.25, label='Accuracy')
    ax.bar(X + 0.25, [k_f1_score, svm_f1_score, ffn_f1], color='g', width=0.25, label='F1 Score')
    ax.set_xticks(X, ['KNN', 'SVM', 'FFN'])
    ax.legend()
    fig.tight_layout()
    plt.show()



