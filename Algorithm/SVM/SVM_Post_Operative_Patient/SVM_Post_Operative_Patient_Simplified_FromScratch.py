## SVM Post Operative Patient From Scratch (using cvxopt and simplified dataset) Version
#
# Author: David Lee
# Create Date: 2018/10/20
#
# Detail:
#   Total Data = 90
#   Training Data : Testing Data = 7 : 3
#   Missing Value : Yes

import numpy as np
import pandas as pd # Read csv
import cvxopt

from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number
from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

# Hide cvxopt output
#cvxopt.solvers.options['show_progress'] = False

## Kernels
def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f

# SVM
class SVM:
    def __init__(self, C=1.0, power=4, gamma=None, kernel='linear', coef=4):
        self.__C = C # Penalty parameter C of the error term
        self.__power = power # The degree of the polynomial kernel
        self.__gamma = gamma # Used in the rbf kernel function
        if kernel == 'rbf':
            kernel = rbf_kernel
        elif kernel == 'poly':
            kernel = polynomial_kernel
        else:
            kernel = linear_kernel
        self.__kernel = kernel # Specifies the kernel type to be used in the algorithm
        self.__coef = coef # Bias term used in the polynomial kernel function

        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.__gamma:
            self.__gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.__kernel = self.__kernel(
            power=self.__power,
            gamma=self.__gamma,
            coef=self.__coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.__kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.__C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.__C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.__kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.__kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        y_test = np.array(y_test)
        prediction = self.predict(X_test)
        return sum(prediction == y_test)/len(y_test)


def loadData(path):
    inputData = pd.read_csv(path)

    # Deal with missing value (8th column: 'COMFORT')
    inputData = inputData.fillna(round(inputData.mean()))
    
    # Transform 'string' into class number
    Labels = [
        ['high', 'mid', 'low'],
        ['high', 'mid', 'low'],
        ['excellent', 'good', 'fair', 'poor'],
        ['high', 'mid', 'low'],
        ['stable', 'mod-stable', 'unstable'],
        ['stable', 'mod-stable', 'unstable'],
        ['stable', 'mod-stable', 'unstable'],
        [str(x) for x in range(21)], # Skip
        ['I', 'S', 'A']
    ]

    le = LabelEncoder()

    dataTemp = np.mat(np.zeros((len(inputData), len(inputData.columns))))
    for colIdx in range(len(inputData.columns)):
        if colIdx != 7:
            le.fit(Labels[colIdx])
            dataTemp[:, colIdx] = np.mat(le.transform(inputData.iloc[:, colIdx])).T
        else:
            # Skip number attribute
            dataTemp[:, colIdx] = np.mat(inputData.iloc[:, colIdx]).T

    num_data = np.array(dataTemp[:, :-1])
    num_label = np.array(dataTemp[:, -1])
    data_train, data_test, label_train, label_test = train_test_split(num_data, num_label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainSVM(data_train, label_train):
    clf = SVM()
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test)))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/post-operative-binary.csv')

    # Train Model
    SVM_model = trainSVM(data_train, label_train)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, SVM_model)))

    # Evaluate Model
    evaluateModel(data_test, label_test, SVM_model)

if __name__ == '__main__':
    main()
