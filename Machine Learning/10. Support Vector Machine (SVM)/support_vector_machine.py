print("Support Vector Machine")
# z = x^2 + y^2
# z is a transformation
#importing iris dataset from sklearn.datasets
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
#get the features of the dataset
iris.feature_names
#get the target of the dataset
iris.target_names

#convert dataset into dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
#create a target colum
df['target'] = iris.target
df.head()

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
#lambada is function or transformation, which transforms the target value to the
#corresponding flower name, for this transformation we use apply function
# target value 0 is converted into setosa
# target value 1 is converted into versicolor
# target value 2 is converted into virginica
df.head()
df0.head()# 0 for setosa
df1.head()# 1 for versicolor
df2.head()# 2 for virginica
# creating the graphs for better visualization
import matplotlib.pyplot as plt
# **Sepal length vs Sepal Width (Setosa vs Versicolor)**
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# **Petal length vs Pepal Width (Setosa vs Versicolor)**
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

# **Train Using Support Vector Machine (SVM)**
from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#find the length of the model
len(X_train)
len(y_train)
#print out the first 5 rows of X_train and y_train
X_train.head()
y_train.head()

from sklearn.svm import SVC
#create an object to train.
model = SVC()
#train the model fit function
model.fit(X_train, y_train)

#check the score of the trained model
acc = model.score(X_test, y_test)
#print the accuracy of the test data
print("Accuracy of the test data is: ",acc*100,"%")
#prediction for seatosa
model.predict([[4.8,3.0,1.5,0.3]])
#moel.predict([[new sepal length (cm),new sepal width (cm),new petal length (cm),new petal width (cm)]])
#prediction for versicolor
model.predict([[6.0,2.9,4.5,1.5]])
#moel.predict([[new sepal length (cm),new sepal width (cm),new petal length (cm),new petal width (cm)]])
#prediction for virginica
model.predict([[6.0,3.4,4.5,2.8]])
#moel.predict([[new sepal length (cm),new sepal width (cm),new petal length (cm),new petal width (cm)]])
# **Tune parameters**
# **1. Regularization (C)**
model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)
model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)
# **2. Gamma**
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)
# **3. Kernel**
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)
model_linear_kernal.score(X_test, y_test)