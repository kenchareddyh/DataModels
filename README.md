# DataModels
This is a small collection of work reproducing the framework detailed in the MIT paper: Datamodels: Predicting Predictions from Training Data

See the paper https://arxiv.org/abs/2202.00622 for more details.

A datamodel is a parametrized function that takes in a fixed example x and any subset of S(the whole training set) as inputs and is able to predict the outcome of x each time. The datamodel is trained on two points. 
1) The value F(x) which is whatever learning algorithm we decided to use trained on some random subset of S(the whole training set), S' and evaulated on a fixed value X
2) A binary vector(0 or 1) that is the same length as S that specifies which examples in the training set are currently being used to train F and evaulate on X(1 if example is in S' and 0 otherwise)
   
This representation provides a unique and simple way to study the quality of the training data such as identifying brittle predictions and understanding train-test leakage.

In the notebook above I reproduced the framework detailed in the paper using several different classification aglorithms for F (Polynomial Regression, KNN, and Nerual Nets) and observed the accuracy of the linear datamodels on various datasets. 

I used the following datasets:
position-salaries dataset: https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv
iris dataset: https://archive.ics.uci.edu/dataset/53/iris
breast cancer dataset: https://archive.ics.uci.edu/dataset/14/breast+cancer
cifar-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
