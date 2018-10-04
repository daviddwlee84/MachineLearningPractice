## Decition Tree Page Blocks Classification Scikit Learn Version
#
# Author: David Lee
# Create Date: 2018/10/3
#
# Detail:
#   Total Data = 5473
#   Training Data : Testing Data = 7 : 3

import numpy as np
import pandas as pd # Read csv

from sklearn.model_selection import train_test_split # Split training and testing data
from sklearn import metrics # Evaluate model

from datetime import datetime # Calculate training time

# A Question is used to partition a dataset
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    # Compare the feature value in an example to the feature value in this question
    def match(self, example):
        val = example[self.column]
        return val >= self.value
    
    def __repr__(self):
        return "Is Attribute[%s] >= %s" % (self.column, self.value)

# A Leaf node classifies data
class Leaf:
    def __init__(self, rows):
        self.predictions = self.__class_counts(rows)

    # Counts the number of each type of example in a dataset
    def __class_counts(self, rows):
        counts = {}  # A dictionary of label -> count.
        for row in rows:
            label = row[-1] # The label is the last column
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

# A Decision Node asks a question
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# A Decision Tree Classifier based on CART algorithm
class CARTDecisionTreeClassifier:
    # Counts the number of each type of example in a dataset
    def __class_counts(self, rows):
        counts = {}  # A dictionary of label -> count.
        for row in rows:
            label = row[-1] # Label is the last column
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    # Partitions a dataset
    def __partition(self, rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    # Calculate the Gini Impurity for a list of rows
    def __gini(self, rows):
        counts = self.__class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    # Information Gain:
    def __info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.__gini(left) - (1 - p) * self.__gini(right)

    # Main Algorithm
    # Find the best question to ask
    # Iterating over every feature / value and calculating the Information Gain
    def __find_best_split(self, rows):
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self.__gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = self.__partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self.__info_gain(true_rows, false_rows, current_uncertainty)

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    # Build the tree
    def __build_tree(self, rows):
        gain, question = self.__find_best_split(rows)

        if gain == 0:
            return Leaf(rows)
        
        true_rows, false_rows = self.__partition(rows, question)

        true_branch = self.__build_tree(true_rows)
        false_branch = self.__build_tree(false_rows)

        return Decision_Node(question, true_branch, false_branch)

    def fit(self, training_data, training_label):
        rows = np.hstack((training_data, training_label[:, None])) # Numpy concatenate 2D arrays with 1D array (shape problem)
        self.__root = self.__build_tree(rows)
    
    # Add "mode" for different return from Leaf
    # In detail mode it will return all the number of labels with format of {labels: counts}
    # In oneAns mode it will return the label with the highest probability => In order to support sklearn.metrics
    def __predictNode(self, testDataRow, node, mode='detail'):
        # Base case: Reach a leaf
        if isinstance(node, Leaf):
            if mode == 'detail':
                return node.predictions
            else:
                return max(node.predictions, key=node.predictions.get)
        
        if node.question.match(testDataRow):
            return self.__predictNode(testDataRow, node.true_branch, mode)
        else:
            return self.__predictNode(testDataRow, node.false_branch, mode)

    def predict(self, testing_data, mode='detail'):
        # If only one row of testing data (i.e. Dimension = 1)
        if testing_data.ndim == 1:
            return self.__predictNode(testing_data, self.__root, mode)
        else:
            prediction = []
            for row in testing_data:
                prediction.append(self.__predictNode(row, self.__root, mode))
            return prediction

    def score(self, testing_data, testing_label, mode='detail'):
        if mode == 'detail':
            predict_label_dict = self.predict(testing_data)
            totalRow = len(testing_label)
            accuracy = 0
            for i in range(totalRow):
                total = sum(predict_label_dict[i].values()) * 1.0
                for lbl in predict_label_dict[i].keys():
                    # Probability of correct label
                    if lbl == testing_label[i]:
                        accuracy += predict_label_dict[i][lbl] / total
            return float(accuracy/totalRow)
        else:
            predict_label = self.predict(testing_data, mode='oneAns')
            total = len(testing_label)
            correct = 0
            for i in range(total):
                if predict_label[i] == testing_label[i]:
                    correct += 1
            return float(correct/total)


    def __print_tree(self, node, spacing=" "):
        if isinstance(node, Leaf):
            print(spacing + 'Predict', node.predictions)
            return
        
        print(spacing + str(node.question))

        print(spacing + '--> True')
        self.__print_tree(node.true_branch, spacing + "   ")

        print(spacing + '--> False')
        self.__print_tree(node.false_branch, spacing + "   ")
    
    def visualization(self, spacing=" "):
        self.__print_tree(self.__root, spacing)

def loadData(path):
    inputData = pd.read_csv(path)
    data = np.array(inputData.drop(['label'], 1))
    label = np.array(inputData['label'])
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=87)
    return data_train, label_train, data_test, label_test
    
def trainDecisionTree(data_train, label_train):
    clf = CARTDecisionTreeClassifier()
    clf.fit(data_train, label_train)
    return clf

def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)

def evaluateModel(data_test, label_test, clf):
    print(metrics.classification_report(label_test, clf.predict(data_test, mode='oneAns')))
    print(metrics.confusion_matrix(label_test, clf.predict(data_test, mode='oneAns')))

def main():
    # Load Data
    data_train, label_train, data_test, label_test = loadData('DecisionTree/DecisionTree_Page_Blocks_Classification/page-blocks.csv')

    # Train Model
    startTime = datetime.now()
    DecisionTreeModel = trainDecisionTree(data_train, label_train)
    print('Training time:', str(datetime.now() - startTime))

    # Test Accuracy
    print('Accuracy:', float(testAccuracy(data_test, label_test, DecisionTreeModel)))

    # Evaluate Model
    evaluateModel(data_test, label_test, DecisionTreeModel)

if __name__ == '__main__':
    main()