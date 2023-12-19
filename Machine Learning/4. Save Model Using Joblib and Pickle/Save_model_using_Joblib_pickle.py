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
plt.plot(df.area, reg.predict(df[["area"]]), color = "blue") #plotting the line
plt.show()


reg = linear_model.LinearRegression() #creating an object for linear regression using linear_model package from sklearn
# reg is the model name 
reg.fit(df[["area"]],df.price) #fit the data (training the model with available data set)
#passing the argumnents i,e dataFrame in 2D as x-axis and price as y-axis
#know, It is ready to predict the price.

#Doing prediction
reg.predict([[5000]])
#By giving the new area , it is going to predict the new price







# First Approach
#  Save Model Using Pickle
import pickle
# creates a file in our directory as a file \
# with open ('file_name_to_create','write_mode') as flie: 
# open a file and load(write) the trained model into it , (because we dont want to train the model again)
with open('model_pickle','wb') as f: # f = file , wb is a write mode
    pickle.dump(reg,f)  #dumping the model into the file 
with open('model_pickle','rb') as f: # opening the file as a only read able file,
    mp = pickle.load(f) # mp is an object
mp.predict([[5000]]) # predicted the data with trained model , now we can share this flie to someOne to use , say: here is my trained model know you can ask the question
 
#know if i want to predict the data , just open the trained model using 
with open('model_name','rb') as f: 
    mp = pickle.load(f)

# and know ask the question like mp.predict()
# no need of training the data again
# Second Approach
# Save Model Using joblib
# from sklearn.externals import joblib
import joblib
#joblib directly takes the file
joblib.dump(reg,'model_joblib') # creates a new file (model_joblib) in our directory and dump the file (reg, which we created) into it 
# joblib.dump(model_name_we_trained_(ex:reg),'Flie_name_to_be_created')
mj = joblib.load('model_joblib')
# joblib.load('file_name') # know using mj.predict we can easily predict the data
mj.predict([[5000]])
mj.coef_
mj.intercept_