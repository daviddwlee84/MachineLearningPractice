## VSM Document Similarity From Scratch Version
#
# Author: David Lee
# Create Date: 2018/12/2
#
# Article amount: 3443

import numpy as np
import pandas as pd
from collections import defaultdict

class Similarity:
    def euclidianDistanceSimilarity(self, A, B):
        return 1.0/(1.0 + np.linalg.norm(A - B))
    def cosineSimilarity(self, A, B):
        num = float(A.T*B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        return 0.5 + 0.5 * (num/denom)

class VSM_Model:
    def __init__(self):
        pass
    
    def __toDictionary(self, articleMatrix):
        """
        For each article, calculate a dictionary
        i.e. word id pair
        """
        for article in articleMatrix:
            pass

    def __DictToCorpus(self, dictionary):
        pass

    def calcTfIdf(self):
        pass

    def getSimilarityMat(self, articleMatrix):
        pass


def documentTokenize(document):
    frequencyOfWords = defaultdict(int)
    for line in document:
        tokens = line.strip().split()
        if tokens: # skip empty lines
            # Extract single word
            for token in tokens[1:]:
                word = token[:token.index('/')]
                pos = token[token.index('/')+1:] # part-of-speech
                # remove common words and meaningless words (using a stoplist)            
                if pos not in ('w', 'y', 'u', 'c', 'p', 'k', 'm'):
                    frequencyOfWords[word] += 1
    return frequencyOfWords

def articlesToMatrix(document, freqOfWords):
    articleMatrix = []
    emptyline = 0 # To seperate each article
    wordsWeWant = []
    articleCounter = 0
    for line in document:
        tokens = line.strip().split()
        if tokens:
            emptyline = 0
            for token in tokens:
                word = token[:token.index('/')]
                # remove the words that only appear once in the corpus
                if freqOfWords[word] > 1:
                    wordsWeWant.append(word)
        else:
            emptyline += 1
        # Found an article
        if emptyline == 3:
            articleMatrix.append(wordsWeWant)
            wordsWeWant = []
            articleCounter += 1
    if wordsWeWant:
        articleMatrix.append(wordsWeWant)
        articleCounter += 1
    print('Total articles:', articleCounter)
    return articleMatrix

def documentPreprocessing(path):
    with open(path, 'r') as chinaNews:
        document = chinaNews.readlines()
    # 1. tokenize the documents
    frequencyOfWords = documentTokenize(document)
    # 2. calculate remain word for each article
    return articlesToMatrix(document, frequencyOfWords)