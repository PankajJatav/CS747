
import tensorflow as tf


import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from model_helper import build_model2
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

evaluation_data_dir = "./data/evaluation/"
preprocessing = Preprocessing(dir_loc=evaluation_data_dir, sub_dir='evaluation')
preprocessing.preprocess()

pre_train_data = "./pre/training/"
pre_validation = "./pre/validation/"
pre_evaluation = "./pre/evaluation/"


# printout versions
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")


BATCH_SIZE = 128
IMG_HEIGHT = 227
IMG_WIDTH = 227

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
model = build_model2()
history = model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=50
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1, 2, 1)
plt.plot(range(1, len(acc)+1), acc, label="Accuracy")
plt.plot(range(1, len(acc)+1), val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title("Training and validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(acc)+1), loss, label="Loss")
plt.plot(range(1, len(acc)+1), val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title("Training and validation Loss")

plt.savefig("model2.png")


eval_data_gen = validation_image_generator.flow_from_directory(directory=pre_evaluation,
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=False,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

eval_result = model.evaluate(eval_data_gen)
print("[test loss, test accuracy]:", eval_result)
