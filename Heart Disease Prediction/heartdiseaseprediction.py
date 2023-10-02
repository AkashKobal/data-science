# -*- coding: utf-8 -*-
"""HeartDiseasePrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CzwydABGC8zIhJdQdsDEbnHWGnNG2k7q

**Understanding the Dataset**

•	Age

•	 Sex

•	 chest pain type (4 values)

•	resting blood pressure

•	 serum cholestoral in mg/dl

•	fasting blood sugar > 120 mg/dl

•	resting electrocardiographic results (values 0,1,2)

•	 maximum heart rate achieved

•	exercise induced angina

•	 oldpeak = ST depression induced by exercise relative to rest

•	 the slope of the peak exercise ST segment

•	 number of major vessels (0-3) colored by flourosopy

•	 thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

•	Target 1->Defective Heart, 0 ->Normal Heart

**Importing Libraries**
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""**Reading Data and Analysis**"""

#reading the csv file in the pandas dataframe
hdata=pd.read_csv('/content/heart_disease_data.csv')

#Printing the first 5 rows of the dataset
hdata.head()

#Printing the last 5 rows of the dataset
hdata.tail()

#Finding the shape of the dataset
hdata.shape

#Getting Information of the dataset
hdata.info()

#Checking the missing values
hdata.isnull().sum()

#statistical information about the dataset
hdata.describe()

#checking the distribution of the Target variable
hdata['target'].value_counts()



"""**Splitting the Features(X) and Target(Y)**"""

X=hdata.drop(columns='target',axis=1)
Y=hdata['target']

X

Y



"""**Splitting the Data into Training data and Testing data**"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

print(X_train.shape,X_test.shape)



"""**Model Training & Evaluation - Logistic Regression & Accuracy Score**"""

model=LogisticRegression()

model.fit(X_train,Y_train)

#predict
y_predict=model.predict(X_test)

acc=accuracy_score(Y_test,y_predict)

acc



"""**Building a Predictive System**"""

inputData=(75,0,2,145,233,1,0,150,0,2.3,0,0,1)
input_array_data=np.asarray(inputData)
input_data_reshaped=input_array_data.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
prediction

if (prediction[0]==1):
  print('The Person has a Heart Disease')
else:
  print('The Person does not have Heart Disease')