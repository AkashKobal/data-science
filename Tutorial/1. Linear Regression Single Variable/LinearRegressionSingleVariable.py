# Linear Regression Single Variable
# How to predict home price using Machine Learning.
# We will use Linear Regression to predict the price of a home in the Bengaluru, YNK area.
# Price = m * area + b (m = slope intercept, b = Y intercept)
# area is an independent variable, Price is a dependent variable (depend on x)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# importing the data file using pandas
df = pd.read_csv("Linear_Regression_Single_Variable_(DataSet).csv")
df

%matplotlib inline
import matplotlib.pyplot as plt



# importing the data file using pandas
df = pd.read_csv("Linear_Regression_Single_Variable_(DataSet).csv")

#ploting the scatter plot to get idea, .scatter(df.name_of_the_colum_for_x-axis, df.name_of_the_colum_for_y-axis, aditional feature(color,size,marker))
plt.scatter(df.area, df.price, color = "red", marker="+")
plt.xlabel("area(sq ft)") #labeling the x-axis
plt.ylabel("price(INR)") #labeling the y-axis

reg = linear_model.LinearRegression() #creating an object for linear regression using linear_model package from sklearn
# reg is the model name 
reg.fit(df[["area"]],df.price) #fit the data (training the model with available data set)
#passing the argumnents i,e dataFrame in 2D as x-axis and price as y-axis
#know, It is ready to predict the price.

#Doing prediction
reg.predict([[3300]])
#By giving the new area , it is going to predict the new price
# y = m * x + b
reg.coef_ # to find the coefficient(m)
reg.intercept_ # to find the intercept(b)  
# y = m * x +b 
y = 135.78767123 * 3300 + 180616.43835616432 #3300 is the area which we want to predict the price
#ploting the line using the predicted data(x-axis(df.area),y-axis(reg.predict(df[['area']])))

plt.scatter(df.area, df.price, color = "red", marker="+")
plt.xlabel("area(sq ft)") #labeling the x-axis
plt.ylabel("price(INR)") #labeling the y-axis
plt.plot(df.area, reg.predict(df[["area"]]), color = "blue") #plotting the line
plt.show()#Predicted price of houses with area greater than 1000 sqft is : **<jupyter_code>print
#ploting the line using the formula y = m * x + b
# df without price 
d = pd.read_csv("Linear_Regression_Single_Variable_(DataSet with area only).csv")
d.head(3)
#predicting the data set using the previous data
# previous data set contain area and price, but new data set contain only area , here we are going to predict whole price of  the data set using previous dataset
p = reg.predict(d) 
reg.predict(d)
d['price'] = p #creating a colum price to store or dispaly the data(predicted price data), and assigning the data(pridicted value) to it.
d
#to get the data (export the data in same csv file)
# d.to_csv("Linear_Regression_Single_Variable_(DataSet with area only).csv",index=False)  #index = False to remove index value (which it will defalt add in csv file while exporting)
#Exercise predict the Canada income of the year 2020 using canada_per_capita_income.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
data = pd.read_csv("canada_per_capita_income.csv")
data.head(5)

%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(df.year,df.income,color = "blue", marker="*")
plt.xlabel("area")
plt.ylabel("price")
plt.plot(df.year,reg.predict(df[['year']]),color = "red")
plt.show()
reg = linear_model.LinearRegression()
reg.fit(df[['year']],df.income)
reg.predict([[2020]])