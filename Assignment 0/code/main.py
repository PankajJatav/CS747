import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier


from preprocessing import Preprocessing

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

acc_values = []
k_values = []
train_data = Normalizer().fit(train_data).transform(train_data)
validation = Normalizer().fit(validation).transform(validation)

knn_max_acc=0
k_max_acc= 0

for i in range(1, 100, 2):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_data, train_data_label)
    predicted = model.predict(validation)
    acc = accuracy_score(validation_label, predicted)
    if acc > knn_max_acc:
        k_max_acc = i
    acc_values.append(acc)
    k_values.append(i)
plt.plot(k_values, acc_values)
plt.show()

mydata = []
epoches = [1, 10 , 100, 1000, 10000, 100000, 1000000]
learning_rates = [1, 0.5, 0.1, 0.001, 0.0001, 0.00001, 0.000001]
reg_constants = [10, 1, 0, 0.1, 0.001 , 0.0001, 0.00001, 0.000001]

svm_max_acc = 0
svm_max_acc_hp = []
for epoche in epoches:
    for learning_rate in learning_rates:
        for reg_constant in reg_constants:
            clf = SGDClassifier(max_iter=epoche, tol=1e-6, alpha=reg_constant, learning_rate='constant', eta0=learning_rate)
            clf.fit(train_data, train_data_label)
            predicted = clf.predict(validation)
            acc = accuracy_score(validation_label, predicted)
            if acc > svm_max_acc:
                svm_max_acc_hp = [epoche, learning_rate, reg_constant]
            mydata.append([epoche, learning_rate, reg_constant, acc])
    print(epoche)

head = ["epoche", "learning_rate", "reg_constant", "accuracy_score"]
print(tabulate(mydata, headers=head, tablefmt="grid"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('learning_rate')
ax.set_ylabel('reg_constant')
ax.set_zlabel('accuracy')
img = ax.scatter(list(zip(*mydata))[1], list(zip(*mydata))[2], list(zip(*mydata))[3], c=list(zip(*mydata))[0], cmap=plt.hot())
fig.colorbar(img)
plt.show()

# Eval for test data
from sklearn.metrics import f1_score

evaluation = Normalizer().fit(evaluation).transform(evaluation)

model = KNeighborsClassifier(n_neighbors=k_max_acc)
model.fit(train_data, train_data_label)
knn_predicted = model.predict(evaluation)
k_acc = accuracy_score(evaluation_label, knn_predicted)
k_f1_score = f1_score(evaluation_label, knn_predicted, average='weighted')



clf = SGDClassifier(max_iter=svm_max_acc_hp[0], tol=1e-6, alpha=svm_max_acc_hp[2], learning_rate='constant', eta0=svm_max_acc_hp[1], n_jobs=-1)
clf.fit(train_data, train_data_label)
svm_predicted = clf.predict(evaluation)
svm_acc = accuracy_score(evaluation_label, svm_predicted)
svm_f1_score = f1_score(evaluation_label, svm_predicted, average='weighted')

X = np.arange(2)
fig, ax = plt.subplots()
ax.bar(X + 0.00, [k_acc, svm_acc], color = 'b', width = 0.25, label='Accuracy')
ax.bar(X + 0.25, [k_f1_score, svm_f1_score], color = 'g', width = 0.25, label='F1 Score')
ax.set_xticks(X, ['KNN', 'SVM'])
ax.legend()
fig.tight_layout()
plt.show()
