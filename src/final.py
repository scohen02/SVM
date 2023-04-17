'''
Wesleyan University, COMP 343, Spring 2022
Final Project
Lexie Silverman, Sophie Cohen
'''

# Python modules
import numpy as np
import pandas as pd
import sys
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss

# Project modules
sys.path.insert(0, './../scripts')
import util

import warnings
warnings.filterwarnings("ignore")



################################################################################
# ---------------------------- SVM from scratch ------------------------------ #

def get_subgradient(X, y, w, C):
    ''' Compute subgradient for weight vector w
        Inputs:
            * X: features array
            * y: target array
            * w: weight vector
            * C: hyperparameter
        Output:
            * subgradient of w
    '''
    num_examples = len(y)
    subgrad_w = 0

    for i in range(num_examples):
        x_i = X[i]
        y_i = y[i]

        if (y_i * (np.dot(w, x_i))) < 1:
            subgrad_w += - (y_i * x_i)
        else:
            subgrad_w += 0

    subgrad_w = w + (C * subgrad_w)
    return subgrad_w

def svm_sgd(X, y, C, epochs):
    ''' Perform stochastic gradient descent
        Inputs:
            * X: features array
            * y: target array
            * C: hyperparameter
            * epochs: number of iterations
        Output:
            * w: weight vector
    '''
    num_dimensions = X.shape[1]
    w = np.zeros(num_dimensions)

    for t in range(1, epochs+1):

        learning_rate = 1/t
        X, y = shuffle(X, y, random_state=0)

        subgrad_w = get_subgradient(X, y, w, C)
        w = w - (learning_rate * subgrad_w)

    return w

def get_predictions(X, y, w):
    ''' Make predictions with weight vector from sgd
        Inputs:
            * X: features array
            * y: target array
            * w: weight vector obtained from sgd
        Output:
            * predictions: numpy array of predicted labels
    '''
    predictions = np.array([])
    for i in range(len(y)):
        x_i = X[i]
        y_pred = util.sgn(np.dot(w, x_i))
        predictions = np.append(predictions, [y_pred])
    return predictions

def get_accuracy(y_true, y_pred):
    ''' Compute the accuracy
        Inputs:
            * y_true: array of true target values
            * y_pred: array of predicted target values
        Output:
            * accuracy: obtained from scikit accuracy_score function
    '''
    return accuracy_score(y_true, y_pred)

def get_loss(X, y, w, C):
    ''' Compute the loss
        Inputs:
            * X: features array
            * y: target array
            * w: weight vector
            * C: hyperparameter
        Output:
        * loss: hinge loss + regularization
    '''
    num_examples = len(y)
    hinge_loss = np.array([0 for _ in range(num_examples)])

    for i in range(num_examples):
        hinge_loss[i] = max(0, (1 - (y[i] * np.dot(w, X[i]))))

    regularization = 1/2 * np.sum(w**2)
    loss = regularization + ((1/num_examples)*C * np.sum(hinge_loss))

    return loss

def svm_cross_validation(df, num_folds, features, target, C, epochs):
    ''' Perform cross validation for svm from scratch
        Inputs:
            * df: dataframe
            * num_folds: number of folds for cross-validation
            * features: feature values
            * target: target value
            * C: hyperparameter
            * epochs: number of epochs
        Outputs:
            * accuracy: average accuracy over num_folds
            * loss: average loss over num_folds
    '''
    folds = util.divide_k_folds(df, num_folds)

    validation_accuracies = []
    validation_losses = []

    for i in range(num_folds):
        training = pd.concat(folds[:i] + folds[i+1:])
        validation = folds[i]

        X_train, y_train = util.get_X_y_data(training, features, target)
        X_test, y_test = util.get_X_y_data(validation, features, target)
        w = svm_sgd(X_train, y_train, C, epochs)

        predictions = get_predictions(X_test, y_test, w)
        accuracy = get_accuracy(validation[target], predictions)
        loss = get_loss(X_test, y_test, w, C)

        validation_accuracies.append(accuracy)
        validation_losses.append(loss)

    return sum(validation_accuracies)/num_folds, sum(validation_losses)/num_folds



################################################################################
# ----------------------------- SVM from scikit ------------------------------ #

# Linear
def scikit_svm_linear(X, y, C, epochs):
    ''' Uses scikit's implementation of support vector machines to classify
        linearly separable data points
        Inputs:
            * train_df: training data
            * test_df: testing data
            * features: features values
            * target: target value
        Output:
            * train accuracy: accuracy of predicted train values
            * test accuracy: accuracy of predicted test values
    '''
    clf = SVC(C=C, kernel="linear", max_iter=epochs)
    clf.fit(X, y)
    return clf

# Non-linear
def scikit_svm_non_linear(X, y, C, epochs):
    ''' Uses scikit's implementation of support vector machines to classify
        non-linearly separable data points
        Inputs:
            * train_df: training data
            * test_df: testing data
            * features: features values
            * target: target value
        Output:
            * train accuracy: accuracy of predicted train values
            * test accuracy: accuracy of predicted test values
    '''
    clf = NuSVC(gamma="auto")
    clf.fit(X, y)
    return clf

def scikit_accuracy(clf, X, y):
    ''' Compute the accuracy
        Inputs:
            * clf: scikit svm model
            * X: feature array
            * y: target array
        Output:
            * accuracy: obtained from scikit accuracy_score function
    '''
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)

def scikit_loss(clf, X, y):
    ''' Compute the loss
        Inputs:
            * clf: scikit svm model
            * X: feature array
            * y: target array
        Output:
            * loss: hinge loss
    '''
    y_pred = clf.predict(X)
    return hinge_loss(y, y_pred)

def scikit_svm_cross_validation(df, num_folds, features, target, C, epochs):
    ''' Perform cross validation for svm from scikit
        Inputs:
            * df: dataframe
            * num_folds: number of folds for cross-validation
            * features: feature values
            * target: target value
            * C: hyperparameter
            * epochs: number of epochs
        Outputs:
            * accuracy: average accuracy over num_folds
            * loss: average loss over num_folds
    '''
    validation_accuracies = []
    validation_losses = []
    folds = util.divide_k_folds(df, num_folds)

    for i in range(num_folds):
        training = pd.concat(folds[:i] + folds[i+1:])
        validation = folds[i]

        X_train, y_train = util.get_X_y_data(training, features, target)
        clf = scikit_svm_linear(X_train, y_train, C, epochs)

        X_test, y_test = util.get_X_y_data(validation, features, target)
        accuracy = scikit_accuracy(clf, X_test, y_test)
        loss = scikit_loss(clf, X_test, y_test)

        validation_accuracies.append(accuracy)
        validation_losses.append(loss)

    return sum(validation_accuracies)/num_folds, sum(validation_losses)/num_folds



################################################################################
#----------------------------- SVM evaluation ---------------------------------#

# Load data
df = pd.read_csv('./../data/data_banknote_authentication.csv')
features = {'variance', 'skewness', 'curtosis', 'entropy'}
target = 'class'

xor = pd.read_csv('./../data/xor.csv')
xor_features = {'A','B'}
xor_target = 'C'

# Add bias feature to data
df['bias'] = 1
features.add('bias')

xor['bias'] = 1
xor_features.add('bias')

# Convert 0 labels to -1
df['class'] = df['class'].apply(lambda x: 1 if x > 0 else -1)

# Split data into training and test
train_proportion = 0.70
train_df, test_df = util.split_data(df, train_proportion)
train_xor, test_xor = util.split_data(xor, train_proportion)

# Number of folds to use in cross-validation
num_folds = 5


print("################# Linear SVM from scratch #################")

print("\n----- Cross-validation -----")
best_C = None
best_epoch = None
best_loss = 1000000
C_values = [0.1, 1.0, 10, 100]
num_epochs = [1, 10, 100, 1000]

for epoch in num_epochs:
    for C in C_values:
        accuracy, loss = svm_cross_validation(train_df, num_folds, features, target, C, epoch)
        print("C:", C, ", \t epochs:", epoch, ", \t loss:", loss, ", \t accuracy:", accuracy)

        if loss < best_loss:
            best_loss = loss
            best_C = C
            best_epoch = epoch

print("best C:", best_C, ", best epoch:", best_epoch, ", best loss:", best_loss)

print("\n----- Linear data -----")
X_train, y_train = util.get_X_y_data(train_df, features, target)
X_test, y_test = util.get_X_y_data(test_df, features, target)
w = svm_sgd(X_train, y_train, best_C, best_epoch)

train_predictions = get_predictions(X_train, y_train, w)
train_accuracy = get_accuracy(y_train, train_predictions)
train_loss = get_loss(X_train, y_train, w, best_C)

test_predictions = get_predictions(X_test, y_test, w)
test_accuracy = get_accuracy(y_test, test_predictions)
test_loss = get_loss(X_test, y_test, w, best_C)

print("train accuracy:", train_accuracy, ", train loss:", train_loss)
print("test accuracy:", test_accuracy, ", test loss:", test_loss)


print("\n----- XOR (non-linear) data -----")
X_train, y_train = util.get_X_y_data(train_xor, xor_features, xor_target)
X_test, y_test = util.get_X_y_data(test_xor, xor_features, xor_target)
w = svm_sgd(X_train, y_train, best_C, best_epoch)

train_predictions = get_predictions(X_train, y_train, w)
train_accuracy = get_accuracy(y_train, train_predictions)
train_loss = get_loss(X_train, y_train, w, best_C)

test_predictions = get_predictions(X_test, y_test, w)
test_accuracy = get_accuracy(y_test, test_predictions)
test_loss = get_loss(X_test, y_test, w, best_C)

print("train accuracy:", train_accuracy, ", train loss:", train_loss)
print("test accuracy:", test_accuracy, ", test loss:", test_loss)


print("\n################# Linear SVM from scikit #################")

print("\n----- Cross-validation -----")
best_C = None
best_epoch = None
best_loss = 1000000
C_values = [0.1, 1.0, 10, 100]
num_epochs = [1, 10, 100, 1000]

for epoch in num_epochs:
    for C in C_values:
        accuracy, loss = scikit_svm_cross_validation(train_df, num_folds, features, target, C, epoch)
        print("C:", C, ", \t epochs:", epoch, ", \t loss:", loss, ", \t accuracy:", accuracy)

        if loss < best_loss:
            best_loss = loss
            best_C = C
            best_epoch = epoch

print("best C:", best_C, ", best epoch:", best_epoch, ", best loss:", best_loss)

print("----- Linear data -----")
X_train, y_train = util.get_X_y_data(train_df, features, target)
clf = scikit_svm_linear(X_train, y_train, best_C, best_epoch)
train_accuracy = scikit_accuracy(clf, X_train, y_train)
train_loss = scikit_loss(clf, X_train, y_train)

X_test, y_test = util.get_X_y_data(test_df, features, target)
test_accuracy = scikit_accuracy(clf, X_test, y_test)
test_loss = scikit_loss(clf, X_test, y_test)

print("train accuracy:", train_accuracy, ", train loss:", train_loss)
print("test accuracy:", test_accuracy, ", test loss:", test_loss)


print("\n----- XOR (non-linear) data -----")
X_train, y_train = util.get_X_y_data(train_xor, xor_features, xor_target)
clf = scikit_svm_linear(X_train, y_train, best_C, best_epoch)
train_accuracy = scikit_accuracy(clf, X_train, y_train)
train_loss = scikit_loss(clf, X_train, y_train)

X_test, y_test = util.get_X_y_data(test_xor, xor_features, xor_target)
test_accuracy = scikit_accuracy(clf, X_test, y_test)
test_loss = scikit_loss(clf, X_test, y_test)

print("train accuracy:", train_accuracy, ", train loss:", train_loss)
print("test accuracy:", test_accuracy, ", test loss:", test_loss)


print("\n################# Non-linear SVM from scikit #################")

print("----- XOR (non-linear) data -----")
X_train, y_train = util.get_X_y_data(train_xor, xor_features, xor_target)
clf = scikit_svm_non_linear(X_train, y_train, best_C, best_epoch)
train_accuracy = scikit_accuracy(clf, X_train, y_train)
train_loss = scikit_loss(clf, X_train, y_train)

X_test, y_test = util.get_X_y_data(test_xor, xor_features, xor_target)
test_accuracy = scikit_accuracy(clf, X_test, y_test)
test_loss = scikit_loss(clf, X_test, y_test)

print("train accuracy:", train_accuracy, ", train loss:", train_loss)
print("test accuracy:", test_accuracy, ", test loss:", test_loss)
