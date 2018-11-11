## Automobile From Scratch Version
#
# Author: David Lee
# Create Date: 2018/11/10
#
# PS. Recommendation System

import numpy as np
import pandas as pd # Read csv
from scipy.sparse.linalg import svds # using ARPACK as an eigensolver
from scipy.sparse import csc_matrix

### Recommendation System
# Item-based recommendation engine

## Similarity Measures
# These metrics assumed the data was in column vectors
def euclidianDistanceSimilarity(A, B):
    return 1.0/(1.0 + np.linalg.norm(A - B))

def pearsonCorrelationSimilarity(A, B):
    if len(A) < 3: return 1
    else: return 0.5 + 0.5 * np.corrcoef(A, B, rowvar=0)[0][1]

def cosineSimilarity(A, B):
    num = float(A.T*B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return 0.5 + 0.5 * (num/denom)
######

def standEstimation(dataMat, user, similarityMeasurement, item):
    n_items = np.shape(dataMat)[1]
    simTotal = 0; ratSimTotal = 0
    for j in range(n_items):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = similarityMeasurement(dataMat[overLap, item], dataMat[overLap, j])
        print("The %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal

def svdEstimation(dataMat, user, similarityMeasurement, item):
    n_items = np.shape(dataMat)[1]
    simTotal = 0; ratSimTotal = 0
    U, Sigma, VT = np.linalg.svd(dataMat, full_matrices=False)
    U_new, Sigma_new, VT_new = keepSingularValue(U, Sigma, VT)
    diagonal_mat = np.mat(np.eye(len(Sigma_new)) * Sigma_new) # Create diagonal matrix
    xformedItems = dataMat.T * U_new * diagonal_mat.I # Create transformed items
    for j in range(n_items):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = similarityMeasurement(xformedItems[item, :].T, xformedItems[j, :].T)
        print("The %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas=cosineSimilarity, estMethod=svdEstimation):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return "You rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
########

def loadExData1_1():
    matrix = [[1, 1, 1, 0, 0],
              [2, 2, 2, 0, 0],
              [1, 1, 1, 0, 0],
              [5, 5, 5, 0, 0],
              [1, 1, 0, 2, 2],
              [0, 0, 0, 3, 3],
              [0, 0, 0, 1, 1]]
    return np.mat(matrix)

def loadExData1_2():
    matrix = [[4, 4, 0, 2, 2],
              [4, 0, 0, 3, 3],
              [4, 0, 0, 1, 1],
              [1, 1, 1, 2, 0],
              [2, 2, 2, 0, 0],
              [1, 1, 1, 0, 0],
              [5, 5, 5, 0, 0]]
    return np.mat(matrix)

def textbook_example():
    ## Machine Learning in Action Example
    ex_data = loadExData1_1()
    U, Sigma, VT = np.linalg.svd(ex_data) # This will consume all your RAM
    print(Sigma)

    # Drop the last two value since they're so small to ignore
    Sig3 = np.mat([[Sigma[0], 0, 0],
                   [0, Sigma[1], 0],
                   [0, 0, Sigma[2]]])

    # Reconstruct an apprximation of the original matrix
    approx_mat = U[:, :3] * Sig3 * VT[:3, :]

    print('Original matrix\n', ex_data)
    print('Apprximate matrix by SVD\n', approx_mat)

    # Euclidian Similarity
    print(euclidianDistanceSimilarity(approx_mat[:, 0], approx_mat[:, 4]))
    print(euclidianDistanceSimilarity(approx_mat[:, 0], approx_mat[:, 0]))
    print(pearsonCorrelationSimilarity(approx_mat[:, 0], approx_mat[:, 4]))
    print(pearsonCorrelationSimilarity(approx_mat[:, 0], approx_mat[:, 0]))
    print(cosineSimilarity(approx_mat[:, 0], approx_mat[:, 4]))
    print(cosineSimilarity(approx_mat[:, 0], approx_mat[:, 0]))

    ex_data2 = loadExData1_2()

    # Recommend user 2
    print(recommend(ex_data2, 2, simMeas=euclidianDistanceSimilarity, estMethod=standEstimation))
    print(recommend(ex_data2, 2, simMeas=pearsonCorrelationSimilarity, estMethod=standEstimation))
    print(recommend(ex_data2, 2, simMeas=cosineSimilarity, estMethod=standEstimation))

def loadExData2():
    matrix = [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
              [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
              [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
              [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
              [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
              [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
              [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
              [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
    return np.mat(matrix)

def textbook_example2():
    data = loadExData2()
    # Recommend user 1
    print(recommend(data, 1, simMeas=euclidianDistanceSimilarity, estMethod=svdEstimation))
    print(recommend(data, 1, simMeas=pearsonCorrelationSimilarity, estMethod=svdEstimation))
    print(recommend(data, 1, simMeas=cosineSimilarity, estMethod=svdEstimation))

def loadData(path):
    ratings_matrix = pd.read_csv(path, index_col=0) # the first column is index
    return np.mat(ratings_matrix)

# Keep default 90% of the energy expressed in the matrix
def keepSingularValue(U, Sigma, VT, energy=0.9):
    target = sum(Sigma) * energy
    temp_sum = 0
    for i, sigma in enumerate(Sigma):
        temp_sum += sigma
        if temp_sum > target:
            break
    return U[:, :i], Sigma[:i], VT[:i, :]
    
def main():
    #textbook_example()

    #textbook_example2()

    ## Anonymous Microsoft Web Data
    # Load Data
    data = loadData('Datasets/MS_ratings_matrix.csv')

    # Truncated SVD
    U, Sigma, VT = np.linalg.svd(data, full_matrices=False) # Compute the entire matrix will consume all your RAM
    #U, Sigma, VT = svds(csc_matrix(data, dtype=float), k = 165)
    U_new, Sigma_new, VT_new = keepSingularValue(U, Sigma, VT, energy=0.9)
    print('90%% of the energy expressed in the matrix is the first %d sigma' % len(Sigma_new))

    print(recommend(data, 1, estMethod=svdEstimation))

if __name__ == '__main__':
    main()
