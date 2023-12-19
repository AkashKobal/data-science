print("Logistic Regression (Binary Classification)")
#Logistic Regression is a technique used to solve classification problem.
#It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
#To represent binary / categorical outcome, we use dummy variables.
#In logistic regression, we get a probabilistic value which lies between 0 and 1.
#So, for example, we can take a threshold value 0.5.
#Now, if the probabilistic value is more than 0.5, we can classify it as 1 (or YES),

# sigmoid function
# sigmoid(z) = 1/(1+e^-z)
# e= Eulers number 2.71828

#in linear function we use y = m*x+c
#in logistic regression we use y = 1/(1+e^-z)
#where z = m*x+c
#here z is the linear function 

import pandas as pd
from matplotlib import pyplot as plt 

# %matplotlib inline

df = pd.read_csv("insurance_data.csv")
df.head()
plt.scatter(df.age, df.bought_insurance, marker = "+", color = "red")
#import train_test_split from sklearn.model_selection

from sklearn.model_selection import train_test_split
# using 90% for training and 10% for testing
# after spliting you get four parameters(X_train, X_test, Y_train, Y_test)
# X_train - independent variable for training set
# X_test - independent variable for testing set
# Y_train - dependent variable for training set
# Y_test - dependent variable for testing set

X_train, X_test, Y_train, Y_test = train_test_split(df[["age"]], df.bought_insurance, train_size = 0.9)
X_test
X_train.head()
# import LogisticRegression from sklearn.linear_model 
from sklearn.linear_model import LogisticRegression
#create an object called model to train
model = LogisticRegression()
# traing the model(object) with X_train,Y_train

model.fit(X_train,Y_train)
# know out model is trained.
# it is ready to predict

# now we will predict for X_test
# predict for X_test

model.predict(X_test)

# 1 =  person buy the insurance
# 0 =  person dont buy insurance 
#chech the accuracy of the model
model.score(X_test,Y_test)
# predict the probability
model.predict_proba(X_test)
#test the model
# predict for age 25
model.predict([[25]])

# 0 means dont buy the insurance
# predict for age 20
model.predict([[50]])

# 1 means person buy the insurance