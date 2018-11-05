## SVM MNIST Multi-class Classifier From Scratch Version
#
# Author: David Lee
# Create Date: 2018/11/4
#
# Detail:
#   Total Data = 42000 --> 5000
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv
import random # Random select index

#from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

class BinarySVM:
    def __init__(self, C=1.0, gamma=0, max_iter=10000, tol=0.001, kernel='linear'):
        self.__C = C
        self.__max_iter = max_iter
        self.__tol = tol
        self.__kernel = kernel
        self.__gamma = gamma

        self.__X = None
        self.__y = None
        self.__n_samples = None
        self.__alphas = None
        self.__error_cache = None
        self.__b = 0
        self.__K = None

    def __kernelTrans(self, X, Xi):
        m = np.shape(X)[0]
        K = np.mat(np.zeros((m, 1)))
        if self.__kernel == 'linear':
            K = X * Xi.T
        elif self.__kernel == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - Xi
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * self.__gamma**2))
        else: raise NameError('Unknown kernel')
        return K

    ##### Sequential Minimal Optimization #####
    def __calcEk(self, k):
        fX_k = float(np.multiply(self.__alphas, self.__y).T * self.__K[:, k] + self.__b)
        E_k = fX_k - float(self.__y[k])
        return E_k

    def __selectJrand(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.__n_samples))
        return j

    def __selectJ(self, i, Ei):
        maxK = -1; maxDeltaE = 0; Ej = 0
        self.__error_cache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.__error_cache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i: continue
                Ek = self.__calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:
            j = self.__selectJrand(i)
            Ej = self.__calcEk(j)
        return j, Ej
    
    def __updateEk(self, k):
        Ek = self.__calcEk(k)
        self.__error_cache[k] = [1, Ek]

    def __clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def __innerL(self, i):
        Ei = self.__calcEk(i)
        if (self.__y[i] * Ei < -self.__tol and self.__alphas[i] < self.__C) or \
           (self.__y[i] * Ei > self.__tol and self.__alphas[i] > 0):
            j, Ej = self.__selectJ(i, Ei)
            alphaIold = self.__alphas[i].copy(); alphaJold = self.__alphas[j].copy()

            if self.__y[i] != self.__y[j]:
                L = max(0, self.__alphas[j] - self.__alphas[i])
                H = min(self.__C, self.__C + self.__alphas[j] - self.__alphas[i])
            else:
                L = max(0, self.__alphas[j] - self.__alphas[i] - self.__C)
                H = min(self.__C, self.__alphas[j] + self.__alphas[i])
            if L==H: 
                return 0

            eta = 2.0 * self.__K[i, j] - self.__K[i, i] - self.__K[j, j]
            if eta >= 0:
                return 0

            self.__alphas[j] -= self.__y[j] * (Ei - Ej) / eta
            self.__alphas[j] = self.__clipAlpha(self.__alphas[j], H, L)
            self.__updateEk(j)
            if abs(self.__alphas[j] - alphaJold) < 1e-5:
                return 0

            self.__alphas[i] += self.__y[j] * self.__y[i] * (alphaJold - self.__alphas[j])
            self.__updateEk(i)

            b1 = self.__b - Ei - self.__y[i] * (self.__alphas[i] - alphaIold) * self.__K[i, i] - \
                 self.__y[j] * (self.__alphas[j] - alphaJold) * self.__K[i, j]
            b2 = self.__b - Ej - self.__y[i] * (self.__alphas[i] - alphaIold) * self.__K[i, j] - \
                 self.__y[j] * (self.__alphas[j] - alphaJold) * self.__K[j, j]
            
            if 0 < self.__alphas[i] and self.__C > self.__alphas[i]: self.__b = float(b1)
            elif 0 < self.__alphas[j] and self.__C > self.__alphas[j]: self.__b = float(b2)
            else: self.__b = float((b1 + b2) / 2.0)
            return 1
        else:
            return 0

    def __smoP(self):
        iter_num = 0; entireSet = True; alphaPairsChanged = 0
        while iter_num < self.__max_iter and ((alphaPairsChanged > 0) or entireSet):  
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.__n_samples):
                    alphaPairsChanged += self.__innerL(i)
                iter_num += 1
            else:
                nonBoundIs = np.nonzero((self.__alphas.A > 0) * (self.__alphas.A < self.__C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__innerL(i)
                iter_num += 1
            if entireSet: entireSet = False
            elif alphaPairsChanged == 0: entireSet = True
        return self.__b, self.__alphas

    ############################################

    def __calcWs(self, alphas, X, y):
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * y[i], X[i, :].T)
        return w

    def fit(self, X, y):
        self.__X = np.mat(X)
        self.__y = np.mat(y).T

        n_samples = np.shape(X)[0]
        self.__n_samples = n_samples
        self.__error_cache = np.mat(np.zeros((n_samples, 2)))
        self.__alphas = np.mat(np.zeros((n_samples, 1)))
        self.__K = np.mat(np.zeros((n_samples, n_samples))) # Kernal
        for i in range(n_samples):
            self.__K[:, i] = self.__kernelTrans(self.__X, self.__X[i, :])

        _, alphas = self.__smoP()

        self.__weight = self.__calcWs(alphas, self.__X, self.__y)

        self.__svInd = np.nonzero(alphas.A>0)[0]
        self.support_vectors = self.__X[self.__svInd]
        self.support_vector_labels = self.__y[self.__svInd]

    def predict(self, X):
        X = np.mat(X)
        m = np.shape(X)[0]
        predictions = []
        for i in range(m):
            kernelEval = self.__kernelTrans(self.support_vectors, X[i, :])
            predict = kernelEval.T * np.multiply(self.support_vector_labels, self.__alphas[self.__svInd]) + self.__b
            ### Return original prediction without np.sign()
            predictions.append(float(predict))
        return np.array(predictions)

class OneVsRestSVM:
    def __init__(self, model=BinarySVM, C=1.0, gamma=0, max_iter=10000, tol=0.001, kernel='linear', norm=True):
        self.__model = model
        self.__models = {} # Pair for each class

        self.__C = C
        self.__max_iter = max_iter
        self.__tol = tol
        self.__kernel = kernel
        self.__gamma = gamma
        self.__norm = norm

        self.__yk = {}

    def __dataNormalize(self, X):
        max_x = np.max(X)
        min_x = np.min(X)

        X -= min_x
        X = (X * 2 / (max_x - min_x)) - 1

        return X

    def __submodelTrain(self, sub_model, X, y_k):
        sub_model.fit(X, y_k)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if(self.__norm): X = self.__dataNormalize(X)

        self.__classes = np.unique(y)

        # Construct model class pair
        for label in self.__classes:
            self.__models[label] = self.__model(C=self.__C, max_iter=self.__max_iter, tol=self.__tol, kernel=self.__kernel, gamma=self.__gamma)
            y_temp = y.copy()
            y_temp[y_temp!=label] = -1; y_temp[y_temp==label] = 1
            self.__yk[label] = y_temp
            print("Now training class:", label)
            self.__submodelTrain(self.__models[label], X, y_temp)

    def __submodelPredict(self, sub_model, X):
        return sub_model.predict(X)

    def predict(self, X_test):
        X_test = np.array(X_test)

        # Probability for each label prediction (column => for each label in order)
        pred_prob = np.zeros((np.shape(X_test)[0], self.__classes.size))
        for i, label in enumerate(self.__classes):
            pred_prob[:, i] = self.__submodelPredict(self.__models[label], X_test)

        # Find the column name which has the maximum value for each row
        self.pred_dataframe_ = pd.DataFrame(pred_prob, columns=self.__classes)

        return np.array(self.pred_dataframe_.idxmax(axis=1))
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return sum(y_pred == y) / len(y)


def loadData(path):
    inputData = pd.read_csv(path)

    inputData = inputData.iloc[-5500:, :]

    label = np.array(inputData['label'])
    data = np.array(inputData.drop(['label'], 1))

    # Use a threshold to binarize data
    data[data<=100] = -1
    data[data>100] = 1

    # Use the last 500 sample to train, and the orther 5000 sample to test
    data_train = data[-500:, :]
    data_test = data[:-500, :]
    label_train = label[-500:]
    label_test = label[:-500]

    #data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)

    return data_train, label_train, data_test, label_test
    
def trainSVM(data_train, label_train):
    clf = OneVsRestSVM(C=1, tol=0.001)
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test)))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('Datasets/MNIST.csv')

    # Train Model
    SVM_model = trainSVM(data_train, label_train)

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, SVM_model)))

    # Evaluate Model
    evaluateModel(data_test, label_test, SVM_model)

if __name__ == '__main__':
    main()
