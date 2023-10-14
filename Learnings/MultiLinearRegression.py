print("MultiLinear Regression")
# Formula 
# price = m1 * area + m2 * bedroom + m3 * age + b // ( y = (M1 * X1) + (M2 * X2) + (M3 * X3) + b )
# area,bedroom,age are independent variable
# price is a dependent variable
# m1,m2,m3 are coefficient and b is an intercept

import pandas as pd
import numpy as np
from sklearn import linear_model
df = pd.read_csv("MultiLinear Regression.csv")
df
# filling the missing value (filling the NaN by taking the medain of the dataColumn(bedroom median))
df.bedroom.median() #calculating the median of the column bedroom
# To get only integer value use math module math.floor
import math  # import the library for floor function
median_bedroom = math.floor(df.bedroom.median()) #floor function
median_bedroom
# syntex : df.cloumName_which_we_Want_to_fill = df.cloumName.fillna(value_to_be_filled)
df.bedroom = df.bedroom.fillna(median_bedroom) #filling the missing value with median value
# first we have the data set with one missing value, so we took median of that missing value column and fill the missing value with median.
df.bedroom
# Now we have the data set with no missing value
df
reg = linear_model.LinearRegression() #creating the linear regression module
reg.fit(df[['area','bedroom','age']],df.price) #fit() is a method we use to train the data set, independent variable are written inside 2D in double
# (i,e  area, bedroom, age) are independent variable , price is a dependent variable
# we train independent variable (i,e  area, bedroom, age) and predict the dependent variable(i,e price)
reg.coef_ # M1, M2, M3
reg.intercept_# b
#predicting the room with 3000 sq ft, bedroom 3. and 40 years old 
reg.predict([[3000,3,40]])
# ( y = (M1 * X1) + (M2 * X2) + (M3 * X3) + b )
y = 137.25 * 3000 + -26025 * 3 + -6825* 40 + 383724.9999999998 
y
# we get the same value as we get from the predict method(rounOff value we get here using formula)

#predicting the room with 2500 sq ft, bedroom 4. and 5 years old 
reg.predict([[2500,4,5]])
#Exercise
# predict 2 yr of experiences, 9 test score, 6 interview score
# predict 12 yr of experiences, 10 test score, 10 interview score 

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("MultiLinear Regression(Exercise).csv")
df
from word2number import w2n
# asuming experience as zero 


# syntax : data_frame.column_name = data_frame.column_name.apply(w2n.word_to_num)
df['experience'] = df['experience'].fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)

# converting the words to number because regression is only for numbers
df
#changing the column name

df.rename(columns={"test_score(out of 10)":"test_score"},inplace=True)
df.rename(columns={"interview_score(out 0f 10)":"interview_score"},inplace=True)
df.rename(columns={"salary($":"salary"},inplace=True)

df
import math
median_testScore = math.floor(df.test_score.median())
median_testScore
df.test_score = df.test_score.fillna(median_testScore)
df
reg = linear_model.LinearRegression()
reg.fit(df[["experience","test_score","interview_score"]],df.salary)
reg.coef_
reg.intercept_
reg.predict([[2,9,6]])