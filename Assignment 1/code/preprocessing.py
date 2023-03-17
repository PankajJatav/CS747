import csv
import os
import pathlib
import re
import numpy as np

import tensorflow as tf

class Preprocessing:
    def __init__(self, dir_loc=None):
        self.data_lable = []
        self.dir_loc = dir_loc
        self.data = []

    def preprocess(self):

        data_dir = pathlib.Path(self.dir_loc)
        imgs_path = data_dir.glob('*.jpg')
        for img_path in imgs_path:
            class_label = str(img_path).split('/')[2][0]
            img = tf.keras.preprocessing.image.load_img(img_path)
            image_array = tf.keras.preprocessing.image.img_to_array(img)
            tf_initial_tensor_constant = tf.image.resize(image_array, [32, 32])
            flatted_array = tf.reshape(tf_initial_tensor_constant, [-1])
            self.data.append(flatted_array.numpy())
            self.data_lable.append(int(class_label))

    def save_to_file(self, filename=None):

        if not os.path.exists('./pre'):
            os.makedirs('./pre')

        np.save("./pre/" + filename + '_label.npy', self.data_lable)
        np.save("./pre/"+filename+'.npy', self.data)
