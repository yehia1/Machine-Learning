import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


file = pd.read_csv("Salary_Data.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,1].values


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0 )

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#for predection 
y_pred = lr.predict(x_test)
y_pred_train = lr.predict(x_train)

#plotting the result
#for training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, y_pred_train, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# for test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, y_pred_train, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
