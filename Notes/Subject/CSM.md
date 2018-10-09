# CSM (Conventional and Social Media Movies)

## Dataset

[CSM (Conventional and Social Media Movies) Dataset 2014 and 2015 Data Set](https://archive.ics.uci.edu/ml/datasets/CSM+%28Conventional+and+Social+Media+Movies%29+Dataset+2014+and+2015)

### Data Set Information

Year:2014 and 2015
Source: Twitter,YouTube,IMDB

### Abstract

-|-
-|-
Data Set Characteristics |Multivariate
Attribute Characteristics|Integer
Number of Attributes     |12
Number of Instances      |217
Associated Tasks         |Classification, Regression
Missing Values?          |Yes

### Source

Mehreen Ahmed
Department of Computer Software Engineering
National University of Sciences and Technology (NUST),
Islamabad, Pakistan
mahreenmcs '@' gmail.com

## Result

Paper - [Using Crowd-Source Based Features from Social Media and Conventional Features to Predict the Movies Popularity](https://www.computer.org/csdl/proceedings/smartcity/2015/1893/00/1893a273-abs.html)

Drop the row with missing values
Measure the accuracy of the test subset (20% of instances) => Same as paper
Takes "Genre, Gross, Budget, Screens, Sequel, Ratings" as Conventional Features
and Drop Gross Income as it is not available before release
and Ratings is the one to be predicted.

Accuracy criteria:
model.score() / Paper criteria Accuracy 2

Model                         |Accuracy       |MAE   |MSE   |RMSE
------------------------------|---------------|------|------|------
Linear Regression Scikit Learn|0.0446 ; 0.7955|0.6158|0.6334|0.7958
