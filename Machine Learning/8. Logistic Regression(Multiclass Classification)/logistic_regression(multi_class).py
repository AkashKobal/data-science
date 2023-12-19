print("Logistic Regression (MultiClass Classification)")
#here we are going to build a model which can read and predict the hand written words
# %matplotlib inline
#importing all the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
dir(digits)
#The dir() function returns a list of names in the current local scope. 
# By calling dir(digits), you're listing all the attributes and methods of the 
# digits object. These will include all the attributes listed above, as
#  well as other attributes that the object may have.
digits.data[0]
#to get the numeric data related to image
#to get the imagedata from data set 
plt.gray()
plt.matshow(digits.images[0])#getting the first image i,e 0, and second image as 1
plt.matshow(digits.images[1])
#to get more image(5 images)
for i in range(5):
    plt.matshow(digits.images[i])
#get target variable
digits.target[0:5]
from sklearn.model_selection import train_test_split
#split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(digits.data,digits.target,test_size=0.2,train_size=0.8)
len(X_test)

len(X_train)
from sklearn.linear_model import LogisticRegression
#create a object of logistic regression to train
model = LogisticRegression()

#train the model using .fit()
model.fit(X_train,Y_train)

# X_train is having the image data
# Y_train is having the corresponding numeric data 

model.score(X_test,Y_test)
# lets go with one random image
plt.matshow(digits.images[67])
#let check the corresponding target data
digits.target[67]
# the output is 6 means, for this image corresponding data(number) is six 

# lets predict with our model
model.predict([digits.data[5]])
#predicting from 0 to 5
model.predict(digits.data[0:5])
#our score is around 95% to check where my model is failed we use of confussion 
#matrix
#confussion matrix is a table which is used to show the performance of the
#classification model on a set of test data for which the true values are known
#as ground truth.
#confussion matrix is used to evaluate the performance of the classification model

from sklearn.metrics import confusion_matrix
#get the predicted value
Y_predicted = model.predict(X_test)

cm = confusion_matrix(Y_test,Y_predicted)
# cm = confusion_matrix(actual data, model predicted data)
cm

# for visualizing the above confussion matrix we are going to use sea born
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.title("Confusion Matrix")
# here we are getting the diagonal values as high values, it means our model
# is working fine


#here in above confussion matric my model say image is 2(x axis), but truth value is 3
#(y axis) is 1 time

# same way 3 times model said 1 for the true value 8
# 1 times model said 8 for the true value 1