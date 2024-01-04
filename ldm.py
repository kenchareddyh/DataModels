import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import random as rd
from sklearn.model_selection import train_test_split 

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical  
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

def estimateDataModel_NN(x, x_train, y_train, a):
    
    m = 100
    size = round(a * len(y_train))
    T_x = np.zeros(shape=(m, x_train.shape[0]))
    T_y = np.zeros(shape=(m,1))
    for i in range(m):
        # sample subset si of S where |si|= a|S| 
        #print(size)
        count = 0
        ind = [0] * len(y_train)
        t = []
        s = np.zeros(shape=(size,x_train.shape[1]))
        z = np.zeros(shape=(size,y_train.shape[1]))
        
       
       
        
        # generate random list of indexes
        indexes = rd.sample(range(x_train.shape[0]), size)
        for index in indexes:
            ind[index] = 1
            s[count] = x_train[index]
            z[count] = y_train[index]
            count+=1
                
        
        #train model  
        s = torch.tensor(s, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        model = nn.Sequential(nn.Linear(x_train.shape[1], 100),
                      nn.ReLU(),
                      nn.Linear(100, y_train.shape[1]),
                      nn.Sigmoid())
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        for epoch in range(100):
            pred_y = model(s)
            loss = criterion(pred_y, z)

            model.zero_grad()
            loss.backward()

            optimizer.step()
            
            
        y = model(x)
        
        y = torch.argmax(y)
        
        T_x[i] = ind
        T_y[i] = y
        
     
    lasso = runRegression(T_x,T_y)
    #print(theta.shape)
    return lasso
         
    
def runRegression(T_x,T_y):
    lasso = Lasso(alpha=0.001)
    lasso.fit(T_x, T_y)
    return lasso
            

def __main__():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Convert y_train into one-hot format
    temp = []
    for i in range(len(y_train)):
        temp.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(temp)
    # Convert y_test into one-hot format
    temp = []
    for i in range(len(y_test)):    
        temp.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(temp)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.50, random_state=42)
    
    # verification nn
    #m = 100
    loc = []
    vals = []
    data_models = []
    a = 0.5
    size = round(a * len(y_train))
    corr_count = 0
    for i in range(len(X_test)):
        x = X_test[i]
        x = np.reshape(X_test[0], (1, 784))
        x = torch.tensor(x, dtype=torch.float)
        ldm_nn = estimateDataModel_NN(x, X_train, y_train, a)
        count = 0
        ind = [0] * len(y_train)
        t = []
        s = np.zeros(shape=(size,X_train.shape[1]))
        z = np.zeros(shape=(size,y_train.shape[1]))

        # generate random list of indexes
        indexes = rd.sample(range(X_train.shape[0]), size)
        for index in indexes:
            ind[index] = 1
            s[count] = X_train[index]
            z[count] = y_train[index]
            count+=1


        # linear model prediction
        prediction = ldm_nn.predict([ind])

        # actual model
        #neigh = KNeighborsClassifier(n_neighbors=3)
        #neigh.fit(s, z)

        #print(y_test[i].type)
        actual = y_test[i].argmax(axis=0)

        if int(round(prediction[0], 0)) == int(actual):
            corr_count += 1
        else:
            #print(ind)
            print("Index", i)
            print("Prediction: " + str(prediction[0]))
            print("Actual: " + str(actual))
            loc.append(i)
            vals.append(ind)
        # percentage difference
        #print(ind)
        #print("Prediction: " + str(prediction[0]))
        #print("Actual: " + str(actual[0]))
        #pdiff = (abs(prediction - actual)/actual) * 100
        #print("% diff" + str(pdiff))
        data_models.append(ldm_nn)

        if (i > 50):
            break

    print("Accuracy: " + str(corr_count/len(X_test)))



    
    
if __name__=="__main__":
    main()


