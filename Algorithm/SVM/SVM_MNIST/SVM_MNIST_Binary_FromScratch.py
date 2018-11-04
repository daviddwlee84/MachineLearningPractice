## SVM MNIST Binary Classifier for 0 From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/30
#
# Detail:
#   Total Data = 42000 --> 5000
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv
import random # Random select index

from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

## Currently support
# Kernel: Linear
# Multiclass - Decision Function: OVR
class SVM:
    def __init__(self, C=1.0, gamma=0, max_iter=10000, tol=0.001, kernel='linear'):
        self.__C = C # Penalty parameter C of the error term (slack variable)
        self.__max_iter = max_iter # Hard limit on iterations within solver
        self.__tol = tol # Tolerance for stopping criterion
        self.__kernel = kernel # Specifies the kernel type to be used in the algorithm
        self.__gamma = gamma # For rbf kernel

        # Will be initialized in fit()
        self.__X = None
        self.__y = None
        self.__n_samples = None
        self.__alphas = None # Lagrange Multiplier
        self.__error_cache = None # Error cache shapt(n_sample, 2): isValid, actual error
        self.__b = 0
        self.__K = None # Kernel of data

    # Kernel function => thansfer to higher dimension
    def __kernelTrans(self, X, Xi):
        m = np.shape(X)[0]
        # try:
        #     m, _ = np.shape(X)
        # except:
        #     # If X is only one row
        #     m = 1
        K = np.mat(np.zeros((m, 1)))

        # if m == 1:
        #     X = np.mat(X)
        #     Xi = np.mat(Xi)

        if self.__kernel == 'linear':
            K = X * Xi.T
        elif self.__kernel == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - Xi
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * self.__gamma**2)) # Element-wise division
        else: raise NameError('Unknown kernel')
        return K

    ##### Sequential Minimal Optimization #####
    def __calcEk(self, k):
        # Calculate error
        fX_k = float(np.multiply(self.__alphas, self.__y).T * self.__K[:, k] + self.__b)
        E_k = fX_k - float(self.__y[k])
        return E_k

    def __selectJrand(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.__n_samples))
        return j

    # Select alpha_j index
    def __selectJ(self, i, Ei): # Inner-loop heuristic
        maxK = -1; maxDeltaE = 0; Ej = 0 # Initialize
        self.__error_cache[i] = [1, Ei] # Update errer cache with Ei
        validEcacheList = np.nonzero(self.__error_cache[:, 0].A)[0] # Get non-zero error
        if len(validEcacheList) > 1: # If there is non-zero error
            for k in validEcacheList: # Try to find the maximum Ek
                if k == i: continue # skip Ei
                Ek = self.__calcEk(k) # Calculate Ek
                deltaE = abs(Ei - Ek) # Calculate |Ei-Ek|
                if deltaE > maxDeltaE: # If found maximum delta E
                    # Choose j for maximum step size
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:
            j = self.__selectJrand(i) # Select j randomly
            Ej = self.__calcEk(j) # Calculate Ej
        return j, Ej
    
    def __updateEk(self, k):
        Ek = self.__calcEk(k) # Calculate Ek
        self.__error_cache[k] = [1, Ek] # Update error cache

    def __clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def __innerL(self, i):
        Ei = self.__calcEk(i) # Calculate error Ei
        if (self.__y[i] * Ei < -self.__tol and self.__alphas[i] < self.__C) or \
           (self.__y[i] * Ei > self.__tol and self.__alphas[i] > 0):
            j, Ej = self.__selectJ(i, Ei) # Second-choice heuristic
            alphaIold = self.__alphas[i].copy(); alphaJold = self.__alphas[j].copy() # Preserve old alpha value

            # Calculate upper and lower bound L and H
            if self.__y[i] != self.__y[j]:
                L = max(0, self.__alphas[j] - self.__alphas[i])
                H = min(self.__C, self.__C + self.__alphas[j] - self.__alphas[i])
            else:
                L = max(0, self.__alphas[j] - self.__alphas[i] - self.__C)
                H = min(self.__C, self.__alphas[j] + self.__alphas[i])
            if L==H: 
                #print("L==H")
                return 0

            # Calculate eta
            eta = 2.0 * self.__K[i, j] - self.__K[i, i] - self.__K[j, j]
            if eta >= 0:
                #print("eta>=0")
                return 0

            # Update alpha_j
            self.__alphas[j] -= self.__y[j] * (Ei - Ej) / eta
            self.__alphas[j] = self.__clipAlpha(self.__alphas[j], H, L) # Trim alpha_j
            self.__updateEk(j)
            if abs(self.__alphas[j] - alphaJold) < 1e-5:
                #print("j not moving enough") # The variation of alpha_j is too small
                return 0
            
            # Update alpha_i
            self.__alphas[i] += self.__y[j] * self.__y[i] * (alphaJold - self.__alphas[j])
            self.__updateEk(i) # Update Ei to error cache

            # Update b_1 and b_2
            b1 = self.__b - Ei - self.__y[i] * (self.__alphas[i] - alphaIold) * self.__K[i, i] - \
                 self.__y[j] * (self.__alphas[j] - alphaJold) * self.__K[i, j]
            b2 = self.__b - Ej - self.__y[i] * (self.__alphas[i] - alphaIold) * self.__K[i, j] - \
                 self.__y[j] * (self.__alphas[j] - alphaJold) * self.__K[j, j]
            
            # Update b accroding to b_1 and b_2
            if 0 < self.__alphas[i] and self.__C > self.__alphas[i]: self.__b = float(b1)
            elif 0 < self.__alphas[j] and self.__C > self.__alphas[j]: self.__b = float(b2)
            else: self.__b = float((b1 + b2) / 2.0)
            return 1
        else:
            return 0

    def __smoP(self):
        iter_num = 0; entireSet = True; alphaPairsChanged = 0 # Initialize
        # If iterate through entire data set but alphas do not update anymore or exceed maximum iteration => break
        while iter_num < self.__max_iter and ((alphaPairsChanged > 0) or entireSet):  
            alphaPairsChanged = 0
            if entireSet: # Iterate through entire data set
                for i in range(self.__n_samples):
                    alphaPairsChanged += self.__innerL(i) # SMO
                    #print("fullSet, iter: %d i: %d, pairs changed %d" % (iter_num, i, alphaPairsChanged))
                iter_num += 1
            else:
                nonBoundIs = np.nonzero((self.__alphas.A > 0) * (self.__alphas.A < self.__C))[0]
                for i in nonBoundIs: # Iterate through non-bound alphas index
                    alphaPairsChanged += self.__innerL(i)
                    #print("non-bound, iter: %d i: %d, pairs changed %d" % (iter_num, i, alphaPairsChanged))
                iter_num += 1
            if entireSet: entireSet = False # Iterate through entire dataset at the first time then switch mode
            elif alphaPairsChanged == 0: entireSet = True # But if alpha doesn't change anymore switch back
            #print("iteration number: %d" % iter_num)
        return self.__b, self.__alphas

    ############################################

    def __calcWs(self, alphas, X, y):
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * y[i], X[i, :].T)
        return w

    def fit(self, X, y):
        # Initialize global variables
        self.__X = np.mat(X)
        self.__y = np.mat(y).T

        n_samples = np.shape(X)[0]
        self.__n_samples = n_samples
        self.__error_cache = np.mat(np.zeros((n_samples, 2)))
        self.__alphas = np.mat(np.zeros((n_samples, 1)))
        self.__K = np.mat(np.zeros((n_samples, n_samples))) # Kernal
        for i in range(n_samples):
            self.__K[:, i] = self.__kernelTrans(self.__X, self.__X[i, :])

        # Use SMO to find b and lagrange multipliers (alpha)
        _, alphas = self.__smoP()

        self.__weight = self.__calcWs(alphas, self.__X, self.__y)

        # Extract support vectors
        self.__svInd = np.nonzero(alphas.A>0)[0] # Get indexes of non-zero lagr. multipiers
        # svInd = alphas > 1e-7
        self.support_vectors = self.__X[self.__svInd] # Get the samples that will act as support vectors
        self.support_vector_labels = self.__y[self.__svInd] # Get the corresponding labels

    def predict(self, X):
        X = np.mat(X)
        m = np.shape(X)[0]
        predictions = []
        for i in range(m):
            kernelEval = self.__kernelTrans(self.support_vectors, X[i, :])
            predict = kernelEval.T * np.multiply(self.support_vector_labels, self.__alphas[self.__svInd]) + self.__b
            #print(int(np.sign(predict)))
            predictions.append(int(np.sign(predict)))
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return sum(y_pred == y) / len(y)

def loadData(path):
    inputData = pd.read_csv(path)

    inputData = inputData.iloc[-5000:, :] # Last 5000 data

    label = np.array(inputData['label'])
    data = np.array(inputData.drop(['label'], 1))

    # Use a threshold to binarize data => Nomalization
    # (Improve accuracy from 0.1 to 0.9)
    data[data<=100] = -1
    data[data>100] = 1

    label[label>0] = 1
    label[label==0] = -1

    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)

    # Use last 5000 data as testing set, the rest of them as training set
    # data_train = data[:, :-5000]
    # data_test = data[:, -5000:]
    # label_train = label[:-5000]
    # label_test = label[-5000:]

    return data_train, label_train, data_test, label_test
    
def trainSVM(data_train, label_train):
    clf = SVM(C=1, tol=0.001)
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
