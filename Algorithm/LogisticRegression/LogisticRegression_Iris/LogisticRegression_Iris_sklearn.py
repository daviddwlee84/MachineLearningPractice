## Logistic Regression Iris Multi-class Classifier Scikit Learn Version
#
# Author: David Lee
# Create Date: 2019/3/24
#
# Detail:
#   Total Data = 140 (with 4 feature)
#   Training Data : Testing Data = 7 : 3

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def loadData():
    # These two method are the same
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    X, y = datasets.load_iris(return_X_y=True)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=87)
    return train_X, test_X, train_y, test_y

def trainLogistic(train_X, train_y, multi_class_method='ovr'):
    clf = LogisticRegression(multi_class=multi_class_method)
    clf.fit(train_X, train_y)
    return clf
    
def testAccuracy(model, test_X, test_y):
    return model.score(test_X, test_y)

def main():
    train_X, test_X, train_y, test_y = loadData()
    LogisticModel = trainLogistic(train_X, train_y, multi_class_method='ovr')
    print("Accuracy of Logistic Regression (OVR) is:", testAccuracy(LogisticModel, test_X, test_y))

    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
    # ‘liblinear’ is limited to one-versus-rest schemes.
    # LogisticModel = trainLogistic(train_X, train_y, multi_class_method='multinomial')
    # print("Accuracy of Logistic Regression (Multinomial) is:", testAccuracy(LogisticModel, test_X, test_y))

if __name__ == "__main__":
    main()
