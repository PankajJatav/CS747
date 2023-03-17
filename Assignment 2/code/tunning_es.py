import tensorflow as tf


import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from model import build_model_reg, build_model_es
from preprocessing import Preprocessing


from matplotlib.pyplot import figure

figure(figsize=(18, 6), dpi=100)

from numba import cuda
device = cuda.get_current_device()

train_data_dir = "./data/training/"
preprocessing = Preprocessing(dir_loc=train_data_dir, sub_dir='training')
preprocessing.preprocess()

validation_data_dir = "./data/validation/"
preprocessing = Preprocessing(dir_loc=validation_data_dir, sub_dir='validation')
preprocessing.preprocess()
# preprocessing.save_to_file('validation')

evaluation_data_dir = "./data/evaluation/"
preprocessing = Preprocessing(dir_loc=evaluation_data_dir, sub_dir='evaluation')
preprocessing.preprocess()
# preprocessing.save_to_file('evaluation')

pre_train_data = "./pre/training/"
pre_validation = "./pre/validation/"
pre_evaluation = "./pre/evaluation/"


# printout versions
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")


# Shape of inputs to NN Model
BATCH_SIZE = 128  # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227  # input Shape required by the model
IMG_WIDTH = 227  # input Shape required by the model

train_image_generator = ImageDataGenerator( rescale=1./255)

validation_image_generator = ImageDataGenerator(rescale=1./255)

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(directory=pre_train_data,
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')


val_data_gen = validation_image_generator.flow_from_directory(directory=pre_validation,
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=False,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

model = build_model_es()

acc = []
val_acc = []
loss = []
val_loss = []
x = []
for i in [1, 3, 5, 10, 50]:
    history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=5
    )
    acc.append(history.history["accuracy"][-1])
    val_acc.append(history.history["val_accuracy"][-1])
    loss.append(history.history["loss"][-1])
    val_loss.append(history.history["val_loss"][-1])
    x.append(i)

plt.subplot(1, 2, 1)
plt.plot(x, acc, label="Accuracy")
plt.plot(x, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title("Training and validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(x, loss, label="Loss")
plt.plot(x, val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title("Training and validation Loss")

plt.savefig("es.png")
