## Logistic Regression Iris Multi-class Classifier From Scratch Version
#
# Author: David Lee
# Create Date: 2019/3/25
#
# Detail:
#   Total Data = 140 (with 4 feature)
#   Training Data : Testing Data = 7 : 3

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

# Reminder
# TODO: Multinomial needs new loss function

class sigmoid: # callable sigmoid function class
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return 1/(1+np.exp(-self.x))
    
    def derivative(self):
        return np.exp(self.x)/(1+np.exp(self.x))**2

class LogisticRegression:
    def __init__(self, tol=1e-4, max_iter=100, multi_class='ovr'):
        self.__tolerance = tol
        self.__max_iter = max_iter
        self.__multiclass = False
        self.__alpha = 0.01 # Learning Rate
        self.__classifierLabel = {} # classifier id: postive label
        self.__classifierWeight = [] # classifier weights (in order)
        assert multi_class in ('ovr', 'multinomial', 'auto') # current support ovr
        self.__multi_class_method = multi_class
    
    def getWeight(self):
        """ Debug usage """
        return self.__classifierWeight

    def __testEarlyStop(self, X, y, weight):
        """ Test tolerance """
        h = sigmoid(np.dot(X, weight))()
        error = y - h

        if sum(abs(error)) < self.__tolerance:
            return True
        else:
            return False
 
    def __stochasticGradientAscent(self, weight, X, y, times):
        """ Stochastic Gradient Ascent

            times is used to descent the learning rate (alpha)
        """
        for _ in range(self.__num_data):
            # Update vectors are randomly selected
            i = int(np.random.uniform(0, self.__num_data))
            # Learning rate changes with each iteration
            self.__alpha += 4/(1.0 + i + times)

            h = sigmoid(np.dot(X[i], weight))()
            error = y[i] - h
            weight = weight + self.__alpha * error * X[i]

        return weight

    def __twoClassClassifier(self, X, y):
        weight = np.ones(self.__num_feature) # initial weight

        for i in range(self.__max_iter):
            weight = self.__stochasticGradientAscent(weight, X, y, i)
            
            # if self.__testEarlyStop(X, y, weight): # early stop
            #     print("Early stop at round {} due to error less than tolerance!".format(i))
            #     break

        return weight

    def __train(self, X, y):
        if self.__multiclass:
            for i in range(self.__num_class):
                y_temp = y.copy()
                y_temp[y == self.__class[i]] = 1
                y_temp[y != self.__class[i]] = 0
                self.__classifierLabel[i] = self.__class[i]
                # print("Multi-class training class {} classifier".format(self.__class[i]))
                weight = self.__twoClassClassifier(X, y_temp)
                self.__classifierWeight.append(weight)
        else:
            y_temp = y.copy()
            y_temp[y == self.__class[0]] = 0
            y_temp[y == self.__class[1]] = 1
            weight = self.__twoClassClassifier(X, y_temp)
            self.__classifierWeight.append(weight)
            self.__classifierLabel[0] = self.__class[1]

    def fit(self, X, y):
        """ Fit the model according to the given training data """
        self.__class = np.unique(y)
        self.__num_class = len(self.__class)
        self.__num_data, self.__num_feature = np.shape(X)

        if self.__num_class > 2:
            self.__multiclass = True
            if self.__multi_class_method == 'auto':
                self.__multi_class_method = 'multinomial'
        else:
            if self.__multi_class_method == 'auto':
                self.__multi_class_method = 'ovr'

        self.__train(X, y)

    def predict_proba(self, X):
        """ Probability estimates """
        num_data = len(X)
        if self.__multiclass:
            result = np.zeros((num_data, self.__num_class)) # initialization
            for class_id, clf_weight in enumerate(self.__classifierWeight):
                for i in range(num_data):
                    h = sigmoid(np.dot(X[i], clf_weight))()
                    result[i, class_id] = h
        else:
            result = np.zeros((num_data, 1)) # initialization
            weight = self.__classifierWeight[0]
            for i in range(num_data):
                h = sigmoid(np.dot(X[i], weight))()
                result[i] = h
 
        return result

    def predict(self, X):
        """ Predict class labels for samples in X """
        num_data = len(X)
        probability = self.predict_proba(X)
        result = np.zeros((num_data, 1))
        if self.__multiclass:
            result = np.argmax(probability, axis=1) # ovr
        else:
            result[probability>0.5] = 1
            result[probability<=0.5] = 0
        return result

    def score(self, X, y):
        """ Returns the mean accuracy on the given test data and labels """
        num_data = len(X)
        prediction = self.predict(X)
        accuracy = 0
        for i in range(num_data):
            if y[i] == prediction[i]:
                accuracy += 1

        accuracy /= num_data

        return accuracy

def loadData():
    X, y = datasets.load_iris(return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

def loadDataBinary():
    X_temp, y_temp = datasets.load_iris(return_X_y=True)
    X = X_temp[y_temp!=2]
    y = y_temp[y_temp!=2]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

def trainLogistic(train_X, train_y, multi_class_method='ovr'):
    clf = LogisticRegression(multi_class=multi_class_method)
    clf.fit(train_X, train_y)
    return clf
 
def testAccuracy(model, test_X, test_y):
    return model.score(test_X, test_y)

def main():
    train_X, test_X, train_y, test_y = loadDataBinary()
    LogisticModel = trainLogistic(train_X, train_y)
    print("Accuracy of Binary (only y==0 & y==1) Logistic Regression is:", testAccuracy(LogisticModel, test_X, test_y))

    train_X, test_X, train_y, test_y = loadData()
    LogisticModel = trainLogistic(train_X, train_y, multi_class_method='ovr')
    print("Accuracy of Multi-class Logistic Regression with OVR is:", testAccuracy(LogisticModel, test_X, test_y))
    score = []
    for _ in range(5):
        LogisticModel = trainLogistic(train_X, train_y, multi_class_method='ovr')
        score.append(LogisticModel.score(test_X, test_y))
    print("average of 5:", np.mean(score))

if __name__ == "__main__":
    main()
