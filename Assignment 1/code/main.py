import numpy as np
from matplotlib import pyplot as plt

from preprocessing import Preprocessing
from utils import plot_bar
from sklearn.preprocessing import Normalizer

from Model import Model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import csv

train_data_dir = "./data/training/"
preprocessing = Preprocessing(dir_loc=train_data_dir)
preprocessing.preprocess()
preprocessing.save_to_file('training')

validation_data_dir = "./data/validation/"
preprocessing = Preprocessing(dir_loc=validation_data_dir)
preprocessing.preprocess()

preprocessing.save_to_file('validation')

evaluation_data_dir = "./data/evaluation/"
preprocessing = Preprocessing(dir_loc=evaluation_data_dir)
preprocessing.preprocess()
preprocessing.save_to_file('evaluation')

train_data = np.load('./pre/training.npy', allow_pickle=True)
train_data_label = np.load('./pre/training_label.npy', allow_pickle=True)
validation = np.load('./pre/validation.npy', allow_pickle=True)
validation_label = np.load('./pre/validation_label.npy', allow_pickle=True)
evaluation = np.load('./pre/evaluation.npy', allow_pickle=True)
evaluation_label = np.load('./pre/evaluation_label.npy', allow_pickle=True)

train_data = Normalizer().fit(train_data).transform(train_data)
validation = Normalizer().fit(validation).transform(validation)
evaluation = Normalizer().fit(evaluation).transform(evaluation)

dropout = [False, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
early_stopping = [False, 2, 4, 8, 16, 32, 64]
regularization = [False, 1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

acc = 0
parameteor = []
graphdata = []
history = []
fields = ['Dropout', 'Early Stopping', 'Regularization value', 'Model Acc', 'Model Loss', 'Validation Accuracy', 'Validation Loss']
best_model = None
for d in dropout:
    for e in early_stopping:
        for r in regularization:
            model = Model(dropout=d, reg_value=r, early_stopping=e)
            (model_acc, loss) = model.fit(train_data, train_data_label)
            y_pred = model.eval(validation, validation_label)
            graphdata.append([y_pred[1], d, e, r])
            history.append([d, e, r, model_acc, loss, y_pred[1], y_pred[0]])
            if acc < y_pred[1]:
                acc = y_pred[1]
                parameteor = ([y_pred[1], d, e, r])
                best_model = model
print("===================Final==============================")
print(acc, parameteor)
print(best_model.eval(evaluation, evaluation_label))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Dropout')
ax.set_ylabel('Early_stopping')
ax.set_zlabel('Regularization')
img = ax.scatter(list(zip(*graphdata))[1], list(zip(*graphdata))[2], list(zip(*graphdata))[3], c=list(zip(*graphdata))[0], cmap=plt.hot())
fig.colorbar(img)
plt.show()


with open('output.csv', 'w') as f:
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(history)

# plot_bar()
