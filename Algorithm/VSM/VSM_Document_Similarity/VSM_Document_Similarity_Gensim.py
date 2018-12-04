## VSM Document Similarity Gensim Version
#
# Author: David Lee
# Create Date: 2018/12/1
#
# Article amount: 3443

from collections import defaultdict # Calculate word appear in article
from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
import pandas as pd

def documentTokenize(document):
    frequencyOfWords = defaultdict(int)
    for line in document:
        tokens = line.strip().split()
        if tokens: # skip empty lines
            #print('processing:', tokens[0][:-2])
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

def VSM(articleMatrix):
    dictionary = corpora.Dictionary(articleMatrix) # Transfer to dictionary
    corpus = [dictionary.doc2bow(article) for article in articleMatrix]   # For each article create a bag-of-words

    # Use TF-IDF Model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Similarity Matrix
    tmp_file = get_tmpfile("vsm_similarity")
    similarity = similarities.Similarity(tmp_file, corpus_tfidf, num_features=len(dictionary))

    # Calculate similarity
    similarityMat = similarity[corpus_tfidf]
    return similarityMat

def main():
    articleMatrix = documentPreprocessing('Datasets/Article/199801_clear_1.txt')
    similarityMat = VSM(articleMatrix)
    simMatDF = pd.DataFrame(similarityMat)
    print(simMatDF.shape)
    print(simMatDF.head())
    #simMatDF.to_csv('SimMat.csv') # about 750MB

if __name__ == "__main__":
    main()
