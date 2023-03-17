import pathlib
import sys

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




