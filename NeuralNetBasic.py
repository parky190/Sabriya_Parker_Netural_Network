import plaidml.keras
plaidml.keras.install_backend()
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

def main(filename, filename2):
    # load the dataset
    n = 89
    dataset = pd.read_csv(filename, skiprows=2)
    new_array = np.array(dataset)
    X = new_array[:,1:90]
    y = new_array[:, 90]

    dataset = pd.read_csv(filename2)
    new_array2 = np.array(dataset)
    X_test = new_array2[:, 1:90]
    y_test = new_array2[:, 90]

    model = Sequential()
    model.add(Dense(10, input_dim=n, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, validation_split=0.33, epochs=5, batch_size=10)
    #model.fit(X, y, epochs=5, batch_size=10)
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))
    print(model.get_weights())
    print(type(model.get_weights()))
    print("Model Summary:")
    print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    scores = model.evaluate(X, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # save model and architecture to single file
    model.save("modelbasic.h5")
    print("Saved model to disk")

    print("loading model")
    # load model
    model = load_model('modelbasic.h5')
    # summarize model.
    model.summary()

    # evaluate the model
    score = model.evaluate(X, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    print("finished! :)")

    # make class predictions with the model
    predictions = model.predict_classes(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, predictions)

    for i in range(0, len(predictions)):
        print('%d (expected %d)' % (predictions[i], y_test[i]))

    plot_roc_curve(fpr, tpr)
    print("auc score")
    auc_score = roc_auc_score(y_test, predictions)
    print(auc_score)

    print(classification_report(y_test, predictions))
    print("confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    return dataset


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == '__main__':
    x = main("Training_10000_each.csv", "Testing_1000_each.csv")
    #x = main("mock.csv")