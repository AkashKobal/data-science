print("Decision Tree")
#Entropy : measure of randomness
#Information Gain : measure of decrease in entropy after the dataset is split
#Gini Index : measure of impurity in a dataset
#Gini Index = 1 - (probability of class 1)^2 - (probability of class 2)^2
#Gini Index = 0 for a pure dataset
#Gini Index = 0.5 for a completely random dataset
#Gini Index = 0.5 for a dataset with 2 classes with equal probability
#Gini Index = 0.66 for a dataset with 3 classes with probability 0.5, 0.25 and 0.25
import pandas as pd
df = pd.read_csv("salaries.csv")
df.head()
# dividing the dataset into independent variable and dependent variable (target variable)
# declaring input for independent variable
input = df.drop('salary_more_then_100k',axis='columns')

# declaring target for dependent variable
target = df['salary_more_then_100k']

input.head()
target.head()
#machine learning can understand only the numbers, so converting the ext data into
#numeric using encoder(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
#we have three feature so create three objects
label_company = LabelEncoder()
label_job = LabelEncoder()
label_degree = LabelEncoder()


#creating a new colum in our dataset using fit transform to store the numeric data
#fit_transform is used to fit and transform the data

input['company_n'] = label_company.fit_transform(input['company'])
input['job_n'] = label_job.fit_transform(input['job'])
input['degree_n'] = label_degree.fit_transform(input['degree'])
input
#in this transformed data google is encoded as 2, abc pharm is encoded as 0,
#facebook is encoded as 1,
#sales executive is encoded as 2,bachelor is encoded as 0, masters as 1 an so on..
#droping all the unwanted colum(colum which is transformed to numeric data)
input_n = input.drop(['company','job','degree'],axis='columns')
input_n.head()
#kow we are going to train our classifier
#we are using decision tree classifier
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(input_n,target)  

model.score(input_n,target)
model.predict([[2,2,1]])
#predicting using encoded number
#2 is facebook, 2 is sales executive, 1 is masters
#output is 0 which means salary is not more than 100k

model.predict([[2,0,1]])
#output is 1 which means salary is more than 100k