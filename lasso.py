from sklearn import linear_model
import pandas as pd
import numpy as np

def main(filename, filename2):

    print("loading dataset")
    # load the dataset
    dataset = pd.read_csv(filename, skiprows=2)
    training_array = np.array(dataset)

    X_train = training_array[:, 1:90]
    y_train = training_array[:, 90]

    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    print(clf.coef_)

    #testset = pd.read_csv(filename2)
    #test_array = np.array(testset)
    #X_test = test_array[:, 1:90]
    #y_test = test_array[:, 90]


if __name__ == '__main__':
    x = main("Training_10000_each.csv", "Testing_1000_each.csv")
    #x = main("mock.csv")