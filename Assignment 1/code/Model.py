import tensorflow as tf


class Model:
    def __init__(self, reg_value=False, early_stopping=False, dropout=False):
        layers_units = [128, 64, 32, 8, 2]
        layers_activation = ['relu', 'relu', 'relu', 'relu', 'relu']
        self.early_stopping = early_stopping
        self.ffn = tf.keras.models.Sequential()
        self.batch_size = 128
        self.epochs = 200

        if dropout is not False:
            self.ffn.add(tf.keras.layers.Dropout(dropout))

        for i in range(0, len(layers_units)):
            if reg_value == False:
                self.ffn.add(tf.keras.layers.Dense(
                    units=layers_units[i],
                    activation=layers_activation[i]
                ))
            else:
                self.ffn.add(tf.keras.layers.Dense(
                    units=layers_units[i],
                    activation=layers_activation[i],
                    kernel_regularizer=tf.keras.regularizers.L2(reg_value)
                ))

        self.ffn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.ffn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def fit(self, X, y):
        if self.early_stopping is not False:
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
            history = self.ffn.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=[callback])
        else:
            history = self.ffn.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

        return (history.history['accuracy'][len(history.history['accuracy']) - 1], history.history['loss'][len(history.history['loss']) - 1])

    def eval(self, X_test, y_test):
        results = self.ffn.evaluate(X_test, y_test, batch_size=128)
        return results

    def predict(self, X_test):
        y_pred = self.ffn.predict(X_test)
        y_pred = (y_pred > 0.5)
        return y_pred
