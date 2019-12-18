from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#linear and logistic regression
def main(filename, filename2):
    # load the dataset
    dataset = pd.read_csv(filename, skiprows=2)
    training_array = np.array(dataset)

    X_train = training_array[:, 1:90]
    y_train = training_array[:, 90]

    testset = pd.read_csv(filename2)
    test_array = np.array(testset)
    X_test = test_array[:, 1:90]
    y_test = test_array[:, 90]

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X_train, y_train)  # perform linear regression

    #print(linear_regressor.coef_)

    Y_pred = linear_regressor.predict(X_test)  # make predictions
    #print(Y_pred)
    print("Linear Regression Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))
    print('Linear Regression Variance score: %.2f' % r2_score(y_test, Y_pred))

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    #print(logreg.coef_)

    Y_pred = logreg.predict(X_test)
    #print(Y_pred)
    print("Logistic Regression Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))
    print('Logistic Regression Variance score: %.2f' % r2_score(y_test, Y_pred))

    lin_det(X_train, y_train, X_test, y_test)
    dec_tree(X_train, y_train, X_test, y_test)
    rand_for(X_train, y_train, X_test, y_test)

# linear discriminant analysis
def lin_det(X, Y, Xtest, Ytest):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, Y)
    clf_pred = clf.predict(Xtest)
    #print("Predicted output of linear discriminant analysis")
    #print(clf_pred)
    print("Linear Discriminant Analysis Mean squared error: %.2f" % mean_squared_error(Ytest, clf_pred))
    print('Linear Discriminant Analysis Variance score: %.2f' % r2_score(Ytest, clf_pred))

# decision trees
def dec_tree(X, Y, Xtest, Ytest):
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(X, Y)
    dtc_pred = dtc.predict(Xtest)
    #print("Predicted output of decision tree classifier")
    #print(dtc_pred)
    print("Decision Tree Mean squared error: %.2f" % mean_squared_error(Ytest, dtc_pred))
    print('Decision Tree Variance score: %.2f' % r2_score(Ytest, dtc_pred))

# random forests
def rand_for(X, Y, Xtest, Ytest):
    rf = RandomForestClassifier(n_estimators=30)
    rf.fit(X, Y)
    rf_pred = rf.predict(Xtest)
    #print("Predicted output of random forest classifier")
    #print(rf_pred)
    print("Random Forest Mean squared error: %.2f" % mean_squared_error(Ytest, rf_pred))
    print('Random Forest Variance score: %.2f' % r2_score(Ytest, rf_pred))

if __name__ == '__main__':
    x = main("Training_10000_each.csv", "Testing_1000_each.csv")
    # x = main("mock.csv")