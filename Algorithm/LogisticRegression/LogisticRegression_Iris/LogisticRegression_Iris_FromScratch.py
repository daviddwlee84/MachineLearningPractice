## Logistic Regression Iris Multi-class Classifier From Scratch Version
#
# Author: David Lee
# Create Date: 2019/3/25
#
# Detail:
#   Total Data = 150 (with 4 feature)
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

class softmax:
    def __init__(self, X):
        self.X = X
    
    def __call__(self):
        return (np.exp(self.X.T) / np.sum(np.exp(self.X), axis=1)).T


class LogisticRegression:
    def __init__(self, tol=1e-4, max_iter=1000, multi_class='ovr', minibatches=1, eta=1e-4, l2=0.0, standardlize=False):
        self.__tolerance = tol # not using this currently
        self.__max_iter = max_iter # means "epoch" when multe_class=='multinomial'
        self.__multiclass = False
        self.__eta = eta # Learning Rate
        self.__standardlize = standardlize

        # === for ovr ===
        self.__classifierLabel = {} # classifier id: postive label
        self.__classifierWeight = [] # classifier weights (in order)
        # ===============

        # === for multinomial ===
        self.__minibatches = minibatches
        self.__l2 = l2 # for L2 norm
        # ===============

        assert multi_class in ('ovr', 'multinomial', 'auto') # current support ovr
        self.__multi_class_method = multi_class
    
    def getWeight(self):
        """ Debug usage """
        if self.__multi_class_method == 'ovr':
            return self.__classifierWeight
        elif self.__multi_class_method == ' multinomial':
            return self.__weight, self.__bias

# ======== One-vs-rest Logistic Regression ======== #

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
            self.__eta += 4/(1.0 + i + times)

            h = sigmoid(np.dot(X[i], weight))()
            error = y[i] - h
            weight = weight + self.__eta * error * X[i]

        return weight

    def __getInitWeight(self, method='uniform'):

        assert method in ('uniform', 'balanced')

        if method == 'uniform':
            # Initialize weights with all one
            return np.ones(self.__num_feature)

        if method == 'balanced':
            # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
            limit = 1 / np.sqrt(self.__num_feature)
            return np.random.uniform(-limit, limit, (self.__num_feature,)) 

    def __twoClassClassifier(self, X, y):
        weight = self.__getInitWeight(method='balanced') # initial weight

        for i in range(self.__max_iter):
            weight = self.__stochasticGradientAscent(weight, X, y, i)
            
            # if self.__testEarlyStop(X, y, weight): # early stop
            #     print("Early stop at round {} due to error less than tolerance!".format(i))
            #     break

        return weight

    def __ovrTrain(self, X, y):
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

# ===== Multinomial Logistic Regression (Softmax Regression) ===== #

    # Error function
    def __cross_entropy(self, output, y_target):
        return -1 * np.sum(np.log(output) * (y_target), axis=1)

    # Cost function
    def __cost(self, cross_entropy):
        L2_term = self.__l2 * np.sum(self.__weight ** 2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def __one_hot(self, y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat

    def __yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])

            if shuffle:
                indices = np.random.permutation(indices)
            if n_batches > 1:
                remainder = data_ary.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch

    def __softmaxTrain(self, X, y):
        y_enc = self.__one_hot(y=y, n_labels=self.__num_class)

        for _ in range(self.__max_iter):
            for idx in self.__yield_minibatches_idx(
                    n_batches=self.__minibatches,
                    data_ary=y,
                    shuffle=True):
                # givens:
                # w_ -> n_feat x n_classes
                # b_  -> n_classes

                # net_input, softmax and diff -> n_samples x n_classes:
                net = np.dot(X[idx], self.__weight) + self.__bias
                softm = softmax(net)()
                diff = softm - y_enc[idx]

                # gradient -> n_features x n_classes
                grad = np.dot(X[idx].T, diff)
                
                # update in opp. direction of the cost gradient
                self.__weight -= (self.__eta * grad +
                            self.__eta * self.__l2 * self.__weight)
                self.__bias -= (self.__eta * np.sum(diff))

            # compute cost of the whole epoch
            net = np.dot(X, self.__weight) + self.__bias
            softm = softmax(net)()
            cross_ent = self.__cross_entropy(output=softm, y_target=y_enc)
            cost = self.__cost(cross_ent)
            self.cost_history.append(cost)

    def fit(self, X, y):
        """ Fit the model according to the given training data """
        self.__class = np.unique(y)
        self.__num_class = len(self.__class)
        self.__num_data, self.__num_feature = np.shape(X)

        if self.__standardlize:
            # use this will lower the performance
            for i in range(self.__num_feature):
                X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

        if self.__num_class > 2:
            self.__multiclass = True
            if self.__multi_class_method == 'auto':
                self.__multi_class_method = 'multinomial'
        else:
            if self.__multi_class_method == 'auto':
                self.__multi_class_method = 'ovr'

        if self.__multi_class_method == 'ovr':
            self.__ovrTrain(X, y)
        elif self.__multi_class_method == 'multinomial':
            # initialize weight and bias
            weights_shape = (self.__num_feature, self.__num_class)
            bias_shape = (1,)
            scale = 0.01
            self.__weight = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
            self.__bias = np.zeros(shape=bias_shape)

            self.cost_history = []

            self.__softmaxTrain(X, y)

    def predict_proba(self, X):
        """ Probability estimates """

        if self.__standardlize:
            # use this will lower the performance
            for i in range(self.__num_feature):
                X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

        num_data = len(X)
        if self.__multi_class_method == 'ovr':
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
        elif self.__multi_class_method == 'multinomial':
            net = np.dot(X, self.__weight) + self.__bias
            result = softmax(net)()
 
        return result

    def predict(self, X):
        """ Predict class labels for samples in X """

        if self.__standardlize:
            # use this will lower the performance
            for i in range(self.__num_feature):
                X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

        probability = self.predict_proba(X)
        if self.__multiclass:
            result = np.argmax(probability, axis=1) # shared between ovr and multinomial
        else:
            num_data = len(X)
            result = np.zeros((num_data, 1))
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

def trainLogistic(train_X, train_y, multi_class_method='ovr', standardlize=True):

    # clf = LogisticRegression(multi_class=multi_class_method, max_iter=100, eta=0.01, standardlize=False)
    # clf = LogisticRegression(multi_class=multi_class_method, max_iter=1000, eta=1e-4, standardlize=False)
    # clf = LogisticRegression(multi_class=multi_class_method, max_iter=100, eta=0.01, standardlize=True)
    # clf = LogisticRegression(multi_class=multi_class_method, max_iter=1000, eta=1e-4, standardlize=True)

    clf = LogisticRegression(multi_class=multi_class_method, standardlize=standardlize)
    clf.fit(train_X, train_y)
    return clf
 
def testAccuracy(model, test_X, test_y):
    return model.score(test_X, test_y)

def main():
    # Binary
    train_X, test_X, train_y, test_y = loadDataBinary()
    LogisticModel = trainLogistic(train_X, train_y)
    print("Accuracy of Binary (only y==0 & y==1) Logistic Regression is:", testAccuracy(LogisticModel, test_X, test_y))

    # OVR
    train_X, test_X, train_y, test_y = loadData()
    NormalLR = trainLogistic(train_X, train_y, multi_class_method='ovr', standardlize=True)
    print("Accuracy of Multi-class Logistic Regression with OVR is:", testAccuracy(NormalLR, test_X, test_y))
    score = []
    for _ in range(5):
        NormalLR = trainLogistic(train_X, train_y, multi_class_method='ovr', standardlize=True)
        score.append(NormalLR.score(test_X, test_y))
    print("average of 5:", np.mean(score))

    # Multinomial
    train_X, test_X, train_y, test_y = loadData() # standardlize data will change the original data
    SoftmaxLR = trainLogistic(train_X, train_y, multi_class_method='multinomial', standardlize=False)
    print("Accuracy of Multi-class Logistic Regression with Multinomial is:", testAccuracy(SoftmaxLR, test_X, test_y))
    score = []
    for _ in range(5):
        SoftmaxLR = trainLogistic(train_X, train_y, multi_class_method='multinomial', standardlize=False)
        score.append(SoftmaxLR.score(test_X, test_y))
    print("average of 5:", np.mean(score))

if __name__ == "__main__":
    main()
