import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(18, 6), dpi=100)

def plot_bar():
    k_acc = 77.2
    svm_acc = 77.8
    k_f1_score = 77.2
    svm_f1_score = 77.8
    ffn_acc = 80.7
    ffn_f1 = 80.7

    cnn_acc = 86.80
    cnn_f1 = 86.80
    cnn_vgg_acc = 88.30
    cnn_vgg_f1= 88.30
    cnn_vgg_aug_acc= 83.40
    cnn_vgg_aug_f1= 83.40

    X = np.arange(6)
    fig, ax = plt.subplots()
    ax.bar(X + 0.00, [k_acc, svm_acc, ffn_acc, cnn_acc, cnn_vgg_acc, cnn_vgg_aug_acc], color='b', width=0.25, label='Accuracy')
    ax.bar(X + 0.25, [k_f1_score, svm_f1_score, ffn_f1, cnn_f1, cnn_vgg_f1, cnn_vgg_aug_f1], color='g', width=0.25, label='F1 Score')
    ax.set_xticks(X, ['KNN', 'SVM', 'FFN', 'OWN CNN', 'VGG_CNN', 'AUG_VGG_CNN'])
    ax.legend()
    fig.tight_layout()
    plt.show()
plot_bar()
