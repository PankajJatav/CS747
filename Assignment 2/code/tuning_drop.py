import tensorflow as tf


import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from model import build_model, build_model_drop
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
                                            # rotation_range=40,
                                            # width_shift_range=0.2,
                                            # height_shift_range=0.2,
                                            # zoom_range=0.2,
                                            # horizontal_flip=True,
                                            # fill_mode='nearest')

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
import keras_tuner as kt
tuner = kt.Hyperband(build_model_drop,
                     objective="val_accuracy",
                     factor=3,
                     directory="logs_dir_drop",
                     project_name="assignment-2")

tuner.results_summary()

tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

drop_data = []

# callbacks at training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get("accuracy")==1.00 and logs.get("loss")<0.03):
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training =True

    def on_train_end(self, logs=None, x=None, y=None):
        print(self, x, y)
        global drop_data
        drop_data.append(
            [
                logs.get('accuracy'),
                logs.get('loss'),
                logs.get('val_accuracy'),
                logs.get('val_loss')
            ]
        )
        print(logs.get('accuracy'),logs.get('loss'))
callbacks = myCallback()

tuner.search(
            train_data_gen,
            validation_data=val_data_gen,
            epochs=5,
            callbacks=[callbacks],
            verbose=2)

best_hps=tuner.get_best_hyperparameters()[0]

print(best_hps)
print(drop_data)
x = [float(y.hyperparameters.values['dropout']) for y in list(tuner.oracle.trials.values())]

indexes = sorted(range(len(x)), key=lambda k: x[k])
drop_data = [drop_data[i] for i in indexes ]
x.sort()

acc = [i[0] for i in drop_data]
val_acc = [i[2] for i in drop_data]
loss = [i[1] for i in drop_data]
val_loss = [i[3] for i in drop_data]
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

plt.savefig("drop.png")
