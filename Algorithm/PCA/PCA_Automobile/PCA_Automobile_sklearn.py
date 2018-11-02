## Automobile Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/11/2
#
# Detail:
#   Total Data = 205

import numpy as np
import pandas as pd # Read csv
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder # Transform 'string' into class number

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