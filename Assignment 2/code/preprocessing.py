import csv
import os
import pathlib
import re
import numpy as np

import tensorflow as tf
import shutil

class Preprocessing:
    def __init__(self, dir_loc=None, sub_dir=None):
        self.data_lable = []
        self.dir_loc = dir_loc
        self.data = []
        self.sub_dir = sub_dir

    def preprocess(self):

        data_dir = pathlib.Path(self.dir_loc)
        imgs_path = data_dir.glob('*.jpg')

        isExist = os.path.exists('./pre/' +self.sub_dir + '/no-food/')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('./pre/' +self.sub_dir + '/no-food/')

        isExist = os.path.exists('./pre/' +self.sub_dir + '/food/')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('./pre/' +self.sub_dir + '/food/')

        for img_path in imgs_path:
            class_label = str(img_path).split('/')[2][0]
            if class_label == '0':
                shutil.copy(img_path, './pre/' + self.sub_dir + '/no-food/'+str(img_path).split('/')[2])
            else:
                shutil.copy(img_path, './pre/' + self.sub_dir + '/food/' + str(img_path).split('/')[2])
