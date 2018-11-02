## Automobile From Scratch Version
#
# Author: David Lee
# Create Date: 2018/11/2
#
# Detail:
#   Total Data = 205

import numpy as np
import pandas as pd # Read csv
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number

class PCA:
    def __init__(self, n_components=None):
        self.__topNfeat = n_components

    # Calculate the covariance matrix for the dataset X
    def __calculate_covariance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

        return np.mat(covariance_matrix, dtype=float)
    
    def __pca(self, dataMat):
        meanVals = np.mean(dataMat, axis=0)
        meanRemoved = dataMat - meanVals # Remove mean

        # Calculate covariance matrix
        covMat = self.__calculate_covariance_matrix(meanRemoved)
        #covMat = np.cov(meanRemoved, rowvar=0)
        
        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigVals, eigVects = np.linalg.eig(covMat)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n components
        eigValIdx = np.argsort(eigVals) # Get the order of the eigenvalues
        eigValIdx = eigValIdx[:-(self.__topNfeat+1):-1] # Sort top N smallest to largest
        reducedEigVects = eigVects[:, eigValIdx]

        lowDDataMat = meanRemoved * reducedEigVects # Transform data into new dimensions
        reconMat = (lowDDataMat * reducedEigVects.T) + meanVals # Reconstruct the original data

        return eigVals, lowDDataMat, reconMat

    def fit(self, data):
        data = np.mat(data)

        if not self.__topNfeat:
            n_samples, n_features = np.shape(data)
            self.__topNfeat = min(n_samples, n_features)
        
        eigVals, lowDMat, reconMat = self.__pca(data)

        self.explained_variance_ratio_ = (eigVals / sum(eigVals))[:self.__topNfeat]
        self.low_dimension_matrix_ = lowDMat
        self.reconstruct_matrix_ = reconMat

    # Return dimensionality reduction matrix of X
    # (Project the data onto principal components)
    # Input (n_samples, n_features) => Return (n_samples, n_components)
    def transform(self, data):
        self.fit(data)
        return self.reconstruct_matrix_

def loadData(path):
    inputData = pd.read_csv(path)

    Make_list = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
         'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
         'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
         'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
    
    # Fill missing value in column "normalized-losses" as its own types car's mean
    for make in Make_list:
        mean_normalized_losses = np.mean(inputData[inputData['make'] == make]['normalized-losses'])
        inputData.update(inputData[inputData['make'] == make]['normalized-losses'].fillna(mean_normalized_losses))
        
    # Drop others whatever contain NA
    inputData = inputData.dropna()

    # Transform 'string' into class number
    Labels = [
        [],
        [],
        Make_list,
        ['diesel', 'gas'],
        ['std', 'turbo'],
        ['four', 'two'],
        ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible'],
        ['4wd', 'fwd', 'rwd'],
        ['front', 'rear'],
        [],
        [],
        [],
        [],
        [],
        ['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'],
        ['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'],
        [],
        ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]

    le = LabelEncoder()

    dataTemp = np.mat(np.zeros((len(inputData), len(inputData.columns))))
    for colIdx in range(len(inputData.columns)):
        # Transform each 'string' label into number
        if Labels[colIdx]:
            le.fit(Labels[colIdx])
            dataTemp[:, colIdx] = np.mat(le.transform(inputData.iloc[:, colIdx])).T
        else:
            dataTemp[:, colIdx] = np.mat(inputData.iloc[:, colIdx]).T

    return dataTemp
    
def tryComponentNumber(data, max_principlal_component):
    pca = PCA(n_components=max_principlal_component)
    pca.fit(data)
    return pca.explained_variance_ratio_

def plotPCAVariance(PCNumbers, percentage_of_variance):
    plt.title('Percentage of total variance contained in the first %d principal components' % PCNumbers)
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.grid(True)
    plt.xticks(range(1, PCNumbers+1))
    plt.plot(np.arange(1, PCNumbers+1), percentage_of_variance*100)

def main():
    # Load Data
    data = loadData('Datasets/imports-85.csv')

    PCNumbers = 20
    ShowFirstPC = 7

    # Train Model
    percentage_of_variance = tryComponentNumber(data, PCNumbers)

    print('%% Variance for the first %d principle component:\n' % ShowFirstPC, np.round(percentage_of_variance*100, 2))
    print('%% Cumulative for the first %d principle component:\n' % ShowFirstPC, [sum(np.round(percentage_of_variance*100, 2)[:i]) for i in range(PCNumbers)])

    # Plot PC number vs. Variance
    plotPCAVariance(PCNumbers, percentage_of_variance)
    plt.show()


if __name__ == '__main__':
    main()