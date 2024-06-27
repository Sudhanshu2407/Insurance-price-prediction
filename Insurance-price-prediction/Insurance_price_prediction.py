#Importing the important libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

#-----------------------------------------------
#Now we read the dataset.

insurance_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\Insurance-price-prediction\insurance.csv")

#-----------------------------------------------
#Now here we check the null values.

insurance_df.isnull().sum()

#Conclusion: As there is no null value.

#-----------------------------------------------
#Now we check the data types of each column.

insurance_df.dtypes

#Conclusion: Numerical column(age,bmi,children,charges)
          #  categorical column(sex,smoker,region)
          
#-----------------------------------------------
#Now we convert the object datatype to categorical data type.

insurance_df[["sex","smoker","region"]]=insurance_df[["sex","smoker","region"]].astype("category")

#----------------------------------------------
#Now we convert them into numerical values using labelencoder.

#----------------------------------------------
#Here we import labelencoder library.
from sklearn.preprocessing import LabelEncoder

#----------------------------------------------
#Here we create labelencoder model object.
lr=LabelEncoder()

#----------------------------------------------
#Here we convert the column from category to numerical.
for i in ["sex","smoker","region"]:
    insurance_df[i]=lr.fit_transform(insurance_df[i])

#----------------------------------------------
#Now here we get the information of dataset.

insurance_df.info()

#----------------------------------------------
#Here we get the description of dataset.

insurance_df.describe()

#----------------------------------------------
#Now here we decide dependent and independent feature.

x=insurance_df.iloc[:,0:-1].values #Independent features.

y=insurance_df.iloc[:,6].values #Dependent feature.

#----------------------------------------------
#Now here we split the dataset into train and test.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#----------------------------------------------
#Now here we do the scaling of data.

from sklearn.preprocessing import StandardScaler

#-----------------------------------
#Here we create standard scaler model object.
sc=StandardScaler()

#-----------------------------------
#Here we fit and transform the x_train. 
x_train_sc=sc.fit_transform(x_train)

#-----------------------------------
#Here we transform the x_test.
x_test_sc=sc.transform(x_test)

#------------------------------------------------
#Now here we create regression model.

#--------------------------------------------
#Here we import Linear regression model.
from sklearn.linear_model import LinearRegression

#--------------------------------------------
#Here we create linear regression model object.
lr=LinearRegression()

#--------------------------------------------
#Here we train the lr model.
lr.fit(x_train,y_train)

#--------------------------------------------
#Here we predict the values using lr model.
y_pred_lr=lr.predict(x_test)

#--------------------------------------------
#Here we find the accuracy of lr model.
score_lr=lr.score(x_test,y_test)
print(f"The accuracy of lr model is {score_lr}.")

#Conclusion: The accuracy of lr model is 79.98%.

#----------------------------------------------
#Here we train the lr model on scaled data.

#---------------------------------------
#Here we create linear regression model object.
lr1=LinearRegression()

#---------------------------------------
#Here we train the lr1 model.
lr1.fit(x_train_sc,y_train)

#---------------------------------------
#Here we predict the values using lr1 model.
y_pred_lr1=lr1.predict(x_test_sc)

#---------------------------------------
#Here we find the accuracy of lr1 model.
score_lr1=lr1.score(x_test_sc,y_test)
print(f"The accuracy of lr1 model is {score_lr1}.")

#Conclusion: The accuracy remain same.

#---------------------------------------------------
#Now we try regularization techniques.

#--------------------------------------
#Here we import regularization models.
from sklearn.linear_model import Lasso,Ridge

#--------------------------------------
#Here we create lasso model object.
ls=Lasso()

#--------------------------------------
#Here we train the ls model.
ls.fit(x_train,y_train)

#--------------------------------------
#Here we predict the values using ls model.
y_pred_ls=ls.predict(x_test)

#--------------------------------------
#Here we find the accuracy of ls model.
score_ls=ls.score(x_test,y_test)
print(f"The accuracy of ls model is {score_ls}.")

#Conclusion: The accuracy of ls model is 79.98%.

#------------------------------------------------------
#Here we train the ls1 model on scaled data.

#--------------------------------------
#Here we create lasso model object.
ls1=Lasso()

#--------------------------------------
#Here we train the ls1 model.
ls1.fit(x_train_sc,y_train)

#--------------------------------------
#Here we predict the values using ls1 model.
y_pred_ls1=ls1.predict(x_test_sc)

#--------------------------------------
#Here we find the accuracy of ls1 model.
score_ls1=ls1.score(x_test_sc,y_test)
print(f"The accuracy of ls1 model is {score_ls1}.")

#Conclusion: The accuracy remain same.

#-------------------------------------------------
#Now we use ridge model.

#--------------------------------------
#Here we create ridge model object.
rd=Ridge()

#--------------------------------------
#Here we train the rd model.
rd.fit(x_train,y_train)

#--------------------------------------
#Here we predict the values using rd model.
y_pred_rd=rd.predict(x_test)

#--------------------------------------
#Here we find the accuracy of rd model.
score_rd=rd.score(x_test,y_test)
print(f"The accuracy of rd model is {score_rd}.")

#Conclusion: The accuracy of rd model is 79.95%.

#----------------------------------------------
#Here we train the rd1 model on scaled data.

#--------------------------------------
#Here we create ridge model object.
rd1=Ridge()

#--------------------------------------
#Here we train the rd1 model.
rd1.fit(x_train_sc,y_train)

#--------------------------------------
#Here we predict the values using rd1 model.
y_pred_rd1=rd1.predict(x_test_sc)

#--------------------------------------
#Here we find the accuracy of rd1 model.
score_rd1=rd1.score(x_test_sc,y_test)
print(f"The accuracy of rd1 model is {score_rd1}.")

#Conclusion: The accuracy of rd1 model is 79.98%.

#-------------------------------------------------------
#Now here we use knn model.

#-------------------------------------------
#Here we import knr model.
from sklearn.neighbors import KNeighborsRegressor

#-------------------------------------------
#Here we build knr model object.
knr=KNeighborsRegressor()

#-------------------------------------------
#Here we train the knr model.
knr.fit(x_train,y_train)

#--------------------------------------------
#Here we predict the values using knr model.
y_pred_knr=knr.predict(x_test)

#--------------------------------------------
#Here we find the accuracy of knr model.
score_knr=knr.score(x_test,y_test)
print(f"The accuracy of knr model is {score_knr}.")

#Conclusion: Here the accuracy is very less.

#----------------------------------------------------
#Now we implement knr on scaled data.

#-------------------------------------------
#Here we build knr1 model object.
knr1=KNeighborsRegressor()

#-------------------------------------------
#Here we train the knr1 model.
knr1.fit(x_train_sc,y_train)

#--------------------------------------------
#Here we predict the values using knr1 model.
y_pred_knr1=knr1.predict(x_test_sc)

#--------------------------------------------
#Here we find the accuracy of knr1 model.
score_knr1=knr1.score(x_test_sc,y_test)
print(f"The accuracy of knr1 model is {score_knr1}.")

#Conclusion: The accuracy of knr1 model is 85%.

#-------------------------------------------------------
#Now here we use svr model.

#---------------------------------------
#Here we import the svr library.
from sklearn.svm import SVR

#----------------------------------------
#Here we build svr model object.
svr=SVR()

#----------------------------------------
#Here we train the svr model.
svr.fit(x_train,y_train)

#-----------------------------------------
#Here we predict the values using svr model.
y_pred_svr=svr.predict(x_test)

#-----------------------------------------
#Here we find the accuracy of svr model.
score_svr=svr.score(x_test,y_test)
print(f"The accuracy of svr model is {score_svr}.")

#Conclusion: Here the accuracy decreases to very low.

#--------------------------------------------
#Now we try to implement svr model on scaled data.

#----------------------------------------
#Here we build svr model object.
svr1=SVR()

#----------------------------------------
#Here we train the svr1 model.
svr1.fit(x_train_sc,y_train)

#-----------------------------------------
#Here we predict the values using svr1 model.
y_pred_svr1=svr1.predict(x_test_sc)

#-----------------------------------------
#Here we find the accuracy of svr1 model.
score_svr1=svr1.score(x_test_sc,y_test)
print(f"The accuracy of svr1 model is {score_svr1}.")

#Conclusion: Here also the accuracy is very less.

#-------------------------------------------------
#Now we try to use decision tree model.

#----------------------------------------
#Here we import the dtr library.
from sklearn.tree import DecisionTreeRegressor

#----------------------------------------
#Here we build dtr model object.
dtr=DecisionTreeRegressor()

#----------------------------------------
#Here we train the dtr model.
dtr.fit(x_train,y_train)

#----------------------------------------
#Here we predict the values using dtr model.
y_pred_dtr=dtr.predict(x_test)
                       
#-----------------------------------------
#Here we find the accuracy of dtr model.
score_dtr=dtr.score(x_test,y_test)
print(f"The accuracy of dtr model is {score_dtr}.")

#Conclusion: The accuracy of dtr model is 65%.

#-----------------------------------------------------
#Here we implement the dtr model on scaled data.

#----------------------------------------
#Here we build dtr1 model object.
dtr1=DecisionTreeRegressor()

#----------------------------------------
#Here we train the dtr1 model.
dtr1.fit(x_train_sc,y_train)

#----------------------------------------
#Here we predict the values using dtr1 model.
y_pred_dtr1=dtr1.predict(x_test_sc)
                       
#-----------------------------------------
#Here we find the accuracy of dtr1 model.
score_dtr1=dtr1.score(x_test_sc,y_test)
print(f"The accuracy of dtr1 model is {score_dtr}.")

#Conclusion: The accuracy remain same.

#----------------------------------------------------------
#Now here we use random forest model.

#------------------------------------------
#Here we import the rfr model.
from sklearn.ensemble import RandomForestRegressor

#------------------------------------------
#Now here we build rfr model object.
rfr=RandomForestRegressor()

#------------------------------------------
#Here train the rfr model.
rfr.fit(x_train,y_train)

#------------------------------------------
#Here we predict the values using rfr model.
y_pred_rfr=rfr.predict(x_test)

#------------------------------------------
#Here we find the accuracy of rfr model.
score_rfr=rfr.score(x_test,y_test)
print(f"The accuracy of rfr model is {score_rfr}.")

#Conclusion: The accuracy of rfr model is 87.63%.

#----------------------------------------------------
#Now here we implement rfr model on scaled data.

#------------------------------------------
#Here we build rfr model object.
rfr1=RandomForestRegressor()

#------------------------------------------
#Here train the rfr1 model.
rfr1.fit(x_train_sc,y_train)

#------------------------------------------
#Here we predict the values using rfr1 model.
y_pred_rfr1=rfr1.predict(x_test_sc)

#------------------------------------------
#Here we find the accuracy of rfr1 model.
score_rfr1=rfr1.score(x_test_sc,y_test)
print(f"The accuracy of rfr1 model is {score_rfr1}.")

#Conclusion: The accuracy remain same.

#-------------------------------------------------------
#Now we do some visualization on the dataset.

#-----------------------------------------------
#Here we use countplot for prce based on sex.
sns.countplot(data=insurance_df,x="sex")

#Conclusion: Here we seen that both male and female get almost equal number of insurance.

#-----------------------------------------------
#Here we use dist plot to get the distribution of charges.
sns.distplot(insurance_df["charges"])

#Conclusion: As the charge is log normally distributed i.e right skewed .

#-----------------------------------------------
#Now here we use bar plot.

#---------------------------------------
#Here Sex Vs Charges
sns.barplot(data=insurance_df,x="sex",y="charges")
plt.title("Sex vs charges")
plt.xlabel("Sex")
plt.ylabel("Charges")
plt.show()

#Conclusion: Here man has more charges than women.

#---------------------------------------
#Here children Vs Charges
sns.barplot(data=insurance_df,x="children",y="charges")
plt.title("children vs charges")
plt.xlabel("Children")
plt.ylabel("Charges")
plt.show()

#Conclusion: When the number of children is 3 and 4 ,the charges is maximum and when no. of children is 6 it is minimum.

#---------------------------------------
#Here smoker Vs Charges
sns.barplot(data=insurance_df,x="smoker",y="charges")
plt.title("smoker vs charges")
plt.xlabel("smoker")
plt.ylabel("Charges")
plt.show()

#Conclusion: If the person is smoker than the charges is thrice than the person is no smoker.

#---------------------------------------
#Here region Vs Charges
sns.barplot(data=insurance_df,x="region",y="charges")
plt.title("region vs charges")
plt.xlabel("region")
plt.ylabel("Charges")
plt.show()

#Conclusion: The charges is maximum for region 2.but almost equal for all region.

#---------------------------------------
#Here we use scatter plot.

#--------------------------------
#Here bmi vs charges. 
sns.scatterplot(data=insurance_df,x="bmi",y="charges")
plt.title("bmi vs charges")
plt.xlabel("bmi")
plt.ylabel("Charges")
plt.show()

#Conclusion: It shows with high bmi,the charges is maximum.

#--------------------------------
#Here age vs charges. 
sns.scatterplot(data=insurance_df,x="age",y="charges")
plt.title("age vs charges")
plt.xlabel("age")
plt.ylabel("Charges")
plt.show()

#Conclusion: Here we seen that for each age the charges are divided into three groups.


#-----------------------------------------------
#Now we save the model.

pickle.dump(rfr,open(r"C:\sudhanshu_projects\project-task-training-course\Insurance-price-prediction\insurance_price_prediction.pkl","wb"))

#-----------------------------------------------
#Now we load the model.

model=pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\Insurance-price-prediction\insurance_price_prediction.pkl","rb"))

#----------------------------------------------
#Now we test the model.

model.score(x_test,y_test)
