print("Training and Testing Data")
#in this data set mileage and age are independent variable and sell price is dependent variable
#in this data set we have to predict the sell price of the car
#we have to split the data into training and testing data
#we have to use linear regression model to predict the sell price of the car
#we have to use train_test_split to split the data into training and testing data
import pandas as pd
df = pd.read_csv("carprices.csv")
df.head(5)
#assigining milage and age to x and sell price to y

x = df[["Mileage","Age(yrs)"]]
y = df["Sell Price($)"]
x.head()
#importing train_test_split from sklearn.model_selection
#split the data into test and training , here we splited data 20% test and 80% for train
#we can also use random_state=10 in train_test_split
#random_state=10 means it will split the data randomly
# after spliting you get four parameters(X_train, X_test, Y_train, Y_test)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

#X_train is the training data of independent variable
#X_test is the testing data of independent variable
#Y_train is the training data of dependent variable
#Y_test is the testing data of dependent variable

X_train
#it choose random data,(run again and clarify)
#to stop the random data should be selected use random state(it freeze the no. of data to be changed)
#X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=10)

len(X_train)
len(X_test)
from sklearn.linear_model import LinearRegression
#using linear model
#creating a linear regression object
#fitting the model to train
clf = LinearRegression()
clf.fit(X_train,Y_train)
clf.predict(X_test)
Y_test
#find the accuracy of the data set using score
clf.score(X_test,Y_test)