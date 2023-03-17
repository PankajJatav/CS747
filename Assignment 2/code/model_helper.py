import tensorflow as tf

def build_model(hp):
    regularization = [0.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    HP_REG = hp.Choice('reg', values=regularization)
    HP_DROP = hp.Float("dropout", 0, 0.9, step=0.1)
    model = tf.keras.models.Sequential()

    # 1st conv
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))

    # 2nd conv
    model.add(tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 4th conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 5th Conv
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))



    # To Flatten layer
    model.add(tf.keras.layers.Flatten())
    # To FC layer 1
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(HP_REG)))


    # add dropout from 0 to 1 with interval of 0.1
    model.add(tf.keras.layers.Dropout(HP_DROP))

    # To FC layer 2
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(HP_REG)))


    # add dropout from 0 to 1 with interval of 0.1
    model.add(tf.keras.layers.Dropout(HP_DROP))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    return model

def build_model_drop(hp):
    HP_DROP = hp.Float("dropout", 0, 0.9, step=0.1)
    model = tf.keras.models.Sequential()

    # 1st conv
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))

    # 2nd conv
    model.add(tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 4th conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 5th Conv
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))



    # To Flatten layer
    model.add(tf.keras.layers.Flatten())
    # To FC layer 1
    model.add(tf.keras.layers.Dense(4096, activation='relu'))


    # add dropout from 0 to 1 with interval of 0.1
    model.add(tf.keras.layers.Dropout(HP_DROP))

    # To FC layer 2
    model.add(tf.keras.layers.Dense(4096, activation='relu'))


    # add dropout from 0 to 1 with interval of 0.1
    model.add(tf.keras.layers.Dropout(HP_DROP))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    return model

def build_model_reg(hp):
    regularization = [0.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    HP_REG = hp.Choice('reg', values=regularization)

    model = tf.keras.models.Sequential()

    # 1st conv
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))

    # 2nd conv
    model.add(tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 4th conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 5th Conv
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))



    # To Flatten layer
    model.add(tf.keras.layers.Flatten())
    # To FC layer 1
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(HP_REG)))

    # To FC layer 2
    model.add(tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(HP_REG)))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    return model

def build_model_es():

    model = tf.keras.models.Sequential()

    # 1st conv
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))

    # 2nd conv
    model.add(tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 4th conv
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # 5th Conv
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))



    # To Flatten layer
    model.add(tf.keras.layers.Flatten())
    # To FC layer 1
    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    # To FC layer 2
    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    return model

def build_model2():
    IMG_WIDTH = 227
    IMG_HEIGHT = 227

    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
    VGG16_MODEL.trainable = False
    model = tf.keras.Sequential()
    model.add(VGG16_MODEL)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model

def resize_and_rescale(image, label):
  IMG_SIZE = 227
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

def augment(image_label, seed):
  IMG_SIZE = 227
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label
